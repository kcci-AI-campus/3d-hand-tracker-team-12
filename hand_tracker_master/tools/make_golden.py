"""Write the C++ tests' reference data (tests/data) from a real validation clip:

  runtime_golden.txt   events (features, validity, camera, capture/arrival) pushed into a HandDirect
                       runtime and its queries, with the expected push results, per-hand camera
                       counts and poses of this reference runtime on the exported ncnn graphs
  litev3_golden.txt    the same for HandLiteV3, from the repository's LiteV3Runtime itself, plus
                       each joint's anchor kind and expected error and each hand's in-view
                       probability; also a 2 s stretch seen by camera 0 only (the joint state)
  features_golden.txt  detected u,v in the slave's UV2 layout plus delay -> the dataset's event
                       features, which make_features must reproduce from models/rig.json

Run from the repository root on the development PC (needs numpy, torch for hand_tracking.data,
and the `ncnn` Python package):

    python hand_tracker_master/tools/make_golden.py
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hand_tracking.data import read_clip                          # noqa: E402
from hand_tracking.runtime import LiteV3Runtime, NcnnGraphs, select_slot_events  # noqa: E402

TIME_UNIT_S, KEEP_MARGIN_S, ARRIVAL_MARGIN_S, SEEN_MIN_JOINTS = .1, .5, .25, 5
UV, ORIGIN, DIRECTION, DELAY = slice(0, 2), slice(2, 5), slice(5, 8), slice(10, 11)


class Event:
    def __init__(self, camera, capture, arrival, tokens, detected):
        self.camera, self.capture, self.arrival, self.tokens, self.detected = camera, capture, arrival, tokens, detected


class ReferenceRuntime:
    """The HandDirect runtime the C++ Runtime mirrors (hand_tracking/direct_runtime.py) plus its
    per-hand camera counts."""

    def __init__(self, directory, fp16=False):
        config = json.loads((Path(directory)/'direct.json').read_text(encoding='utf-8'))['config']
        self.dim, self.per, self.span = config['dim'], config['slots_per_camera'], config['event_span_s']
        self.graphs = NcnnGraphs(directory, ('encoder', 'query'), 1, fp16)
        self.keep = self.span+ARRIVAL_MARGIN_S+KEEP_MARGIN_S
        self.events, self.newest, self.last_arrival = [], [None]*3, None

    def push(self, camera, features, valid, capture, arrival):
        if self.last_arrival is not None and arrival < self.last_arrival:
            raise ValueError('arrival order')
        self.last_arrival = arrival
        if self.newest[camera] is not None and capture <= self.newest[camera]:
            return False
        self.newest[camera] = capture
        raw = np.where(valid[..., None], features[..., :11].astype(np.float64), 0.)
        x = np.concatenate((raw[..., UV]*2-1, raw[..., ORIGIN], raw[..., DIRECTION], raw[..., DELAY]/TIME_UNIT_S,
                            valid[..., None]), -1)
        x = np.where(valid[..., None], x, 0.)
        tokens = np.asarray(self.graphs.run('encoder', x.reshape(2, -1), np.eye(3)[camera]))
        self.events.append(Event(camera, capture, arrival, tokens, valid.sum(-1)))
        while self.events and self.events[0].capture < arrival-self.keep:
            self.events.pop(0)
        return True

    def query(self, time):
        slots = select_slot_events(self.events, time, self.per, self.span)
        present = np.array([e is not None for e in slots])
        tokens = np.stack([e.tokens if e is not None else np.zeros((10, self.dim)) for e in slots])
        ages = np.array([max(time-e.capture, 0.)/TIME_UNIT_S if e is not None else 0. for e in slots])
        pose = np.asarray(self.graphs.run('query', tokens.reshape(-1, self.dim), np.repeat(present, 10),
                                          np.repeat(ages, 10))).reshape(2, 21, 3)
        seen = [len({e.camera for e in slots if e is not None and e.detected[h] >= SEEN_MIN_JOINTS}) for h in range(2)]
        return pose, seen


def numbers(values):
    return ' '.join(repr(float(v)) for v in np.asarray(values, np.float64).reshape(-1))


def litev3_golden(model, clip, epoch, count):
    """HandLiteV3 records: the clip's first count frames (a query after every second one), then
    camera 0 alone for 2 s (the history search runs dry; the state keeps each joint's depth), then
    an outage and one frame back."""
    runtime = LiteV3Runtime(model, fp16=False)
    lines = ['litev3_golden 1 fp16=0 tolerance=1e-3']
    def push(camera, features, valid, capture, arrival):
        accepted = runtime.push(camera, features, valid, capture, arrival)
        lines.append(f'push {camera} {capture!r} {arrival!r} {int(accepted)} {numbers(features)} '
                     + ' '.join(str(int(v)) for v in valid.reshape(-1)))
    def query(time):
        pose = runtime.query(time)
        slots = select_slot_events(runtime.events, time, runtime.config.slots_per_camera, runtime.config.event_span_s)
        seen = [len({e.camera for e in slots if e is not None and e.valid[h].sum() >= SEEN_MIN_JOINTS}) for h in range(2)]
        lines.append(f'query {time!r} {seen[0]} {seen[1]} ' + ' '.join(str(int(k)) for k in runtime.anchor_kind.reshape(-1))
                     + f' {numbers(runtime.in_view_probability)} {numbers(np.log1p(runtime.error_mm))} {numbers(pose)}')
    order = np.argsort(clip['event_arrival'], kind='stable')[:count]
    for n, i in enumerate(order):
        arrival = epoch+float(clip['event_arrival'][i])
        push(int(clip['event_camera'][i]), clip['event_features'][i], clip['event_valid'][i].astype(bool),
             epoch+float(clip['event_capture'][i]), arrival)
        if n % 2:
            query(arrival)
    # Camera 0 alone: its newest frame again every 1/11.7 s.
    last0 = [i for i in order if clip['event_camera'][i] == 0][-1]
    time = runtime.last_arrival
    for step in range(1, 24):
        time = runtime.last_arrival+.0855
        push(0, clip['event_features'][last0], clip['event_valid'][last0].astype(bool), time-.06, time)
        query(time)
    query(time+3.)
    push(1, clip['event_features'][order[-1]], clip['event_valid'][order[-1]].astype(bool), time+3.05, time+3.1)
    query(time+3.1)
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parents[1]
    p.add_argument('--model', default=str(here/'models'/'hand_direct_wrist'))
    p.add_argument('--litev3', default=str(here/'models'/'hand_litev3'))
    p.add_argument('--clip', default=str(ROOT/'exports'/'gigahands_pi3_overlap'/'val'/'clip_00001.npz'))
    p.add_argument('--output', default=str(here/'tests'/'data'))
    p.add_argument('--events', type=int, default=150)
    args = p.parse_args()
    clip = read_clip(args.clip)
    order = np.argsort(clip['event_arrival'], kind='stable')[:args.events]
    epoch = 1000.    # absolute seconds, as a steady clock gives them
    runtime = ReferenceRuntime(args.model)
    # ncnn's fp32 differs across CPUs (ARM NEON vs x86: 1.7e-4 seen on a Pi); 1e-3 world units = 0.4 mm.
    lines = ['runtime_golden 1 fp16=0 tolerance=1e-3']
    def push(camera, features, valid, capture, arrival):
        accepted = runtime.push(camera, features, valid, capture, arrival)
        lines.append(f'push {camera} {capture!r} {arrival!r} {int(accepted)} {numbers(features)} '
                     + ' '.join(str(int(v)) for v in valid.reshape(-1)))
    def query(time):
        pose, seen = runtime.query(time)
        lines.append(f'query {time!r} {seen[0]} {seen[1]} {numbers(pose)}')
    for n, i in enumerate(order):
        features, valid = clip['event_features'][i], clip['event_valid'][i].astype(bool)
        capture, arrival = epoch+float(clip['event_capture'][i]), epoch+float(clip['event_arrival'][i])
        push(int(clip['event_camera'][i]), features, valid, capture, arrival)
        if n % 3 == 2:
            query(arrival)
        if n == 30:                                          # a stale repeat of that frame: ignored
            push(int(clip['event_camera'][i]), features, valid, capture, arrival)
    # Camera outage: nothing within the span, then one camera back.
    last = runtime.last_arrival
    query(last+2.)
    i = order[-1]
    push(int(clip['event_camera'][i]), clip['event_features'][i], clip['event_valid'][i].astype(bool), last+2.05, last+2.1)
    query(last+2.1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out/'runtime_golden.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    lite = litev3_golden(args.litev3, clip, epoch, args.events)
    (out/'litev3_golden.txt').write_text('\n'.join(lite)+'\n', encoding='utf-8')

    # Features: the dataset's event features from the slave-format u,v (-1: not detected).
    rows = ['features_golden 1 tolerance=1e-5']
    for i in order[::5]:
        features, valid = clip['event_features'][i], clip['event_valid'][i].astype(bool)
        uv = np.where(valid[..., None], features[..., :2], -1.).reshape(-1)
        delay = float(clip['event_arrival'][i]-clip['event_capture'][i])
        rows.append(f'event {int(clip["event_camera"][i])} {delay!r} {numbers(uv)} {numbers(features)} '
                    + ' '.join(str(int(v)) for v in valid.reshape(-1)))
    (out/'features_golden.txt').write_text('\n'.join(rows)+'\n', encoding='utf-8')
    print(f'Wrote {len(lines)-1} HandDirect, {len(lite)-1} HandLiteV3 and {len(rows)-1} feature records to {out}')


if __name__ == '__main__':
    main()
