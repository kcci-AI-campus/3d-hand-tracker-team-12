"""Write the golden fixture of the C++ runtime (tests/data): a small random HandLite
exported to ncnn, and golden.txt recording hand_tracking.lite_runtime.LiteRuntime on an
event scenario: every push/query with its outcome and every network call with inputs and
outputs. tests/golden_test.cpp replays the network outputs and checks everything else.

    python cpp/hand_lite/tools/make_golden.py            # from the repository root
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from hand_tracking.config import LiteConfig
from hand_tracking.lite import HandLite
from hand_tracking.lite_export import export_lite
from hand_tracking.lite_runtime import LiteRuntime

ORIGINS = np.array([[-1., -1., .5], [1., -1., .5], [0., 1., 1.]])
FORMAT = 'hand_lite_golden 1'


class Recording:
    """Wraps a LiteRuntime backend and keeps every call of the current operation."""

    def __init__(self, graphs):
        self.graphs, self.calls = graphs, []

    def run(self, name, *arrays):
        output = self.graphs.run(name, *arrays)
        self.calls.append((name, [np.asarray(a, np.float32) for a in arrays], np.asarray(output, np.float32)))
        return output


def tensor_line(array):
    array = np.asarray(array, np.float32)
    return ' '.join([str(array.ndim), *map(str, array.shape), *(f'{v:.9g}' for v in array.ravel())])


def scenario(seed=7):
    """(kind, arguments, expected error) in order: 3 cameras at 17.1 fps with delays, missing
    joints, an empty frame, a stale packet, bad inputs, a 0.4 s two-camera dropout (samples
    only held), camera 1 reporting the hands swapped for 0.15 s (outlier rays), queries and
    a 1.5 s outage (no slot)."""
    rng = np.random.default_rng(seed)
    base, velocity = rng.normal(size=(2, 21, 3))*.1, rng.normal(size=(2, 21, 3))*.3
    base[1] += (0., .3, .4)                                          # hands apart, so a swap is an outlier
    frames = []
    for start, stop in ((0., 1.4), (2.9, 3.4)):                     # outage between the two runs
        for camera, phase in enumerate((0., .02, .04)):
            for capture in np.arange(start+phase, stop, 1/17.1):
                if camera and .8 < capture < 1.2:                   # cameras 1, 2 drop out: samples age (hold)
                    continue
                frames.append((capture+.03+rng.uniform(0, .04), camera, capture))
    frames.sort()
    operations, resumed = [], False
    for index, (arrival, camera, capture) in enumerate(frames):
        if not resumed and capture > 2.5:                          # first frame after the outage
            resumed = True
            operations.append(('query', (arrival-.001,), None))     # no slot within the span
        position = base+velocity*capture
        if camera == 1 and .5 < capture < .65:                     # handedness swapped in camera 1
            position = position[::-1]
        features = np.zeros((2, 21, 14), np.float32)
        direction = position-ORIGINS[camera]+rng.normal(size=position.shape)*.002
        features[..., 0:2] = rng.uniform(size=(2, 21, 2))
        features[..., 2:5] = ORIGINS[camera]
        features[..., 5:8] = direction/np.linalg.norm(direction, axis=-1, keepdims=True)
        features[..., 10] = arrival-capture
        valid = rng.uniform(size=(2, 21)) > .1
        if index == 20:
            valid[:] = False                                        # a frame without detections
        operations.append(('push', (camera, features, valid, capture, arrival), None))
        if index == 30:                                             # stale: older capture arrives now
            operations.append(('push', (camera, features, valid, capture-.2, arrival), None))
        if index == 31:
            operations.append(('push', (-1, features, valid, capture, arrival), 'invalid'))
            operations.append(('query', (arrival-.01,), 'logic'))   # before the latest arrival
        if index % 3 == 2 or .8 < capture < 1.4:                    # every frame of the dropout
            operations.append(('query', (arrival,), None))
        if index % 7 == 5:
            operations.append(('query', (arrival+.013,), None))
    last = frames[-1][0]
    operations.append(('push', (0, features, valid, last+1., last-.5), 'logic'))  # arrival order
    operations.append(('query', (last+.02,), None))
    return operations


def write(runtime, operations, path):
    recorder = Recording(runtime.graphs)
    runtime.graphs = recorder
    lines = [FORMAT]
    for kind, arguments, expected in operations:
        recorder.calls = []
        if kind == 'push':
            camera, features, valid, capture, arrival = arguments
            lines.append(f'op push {camera} {capture:.17g} {arrival:.17g}')
            lines.append(' '.join(f'{v:.9g}' for v in features.ravel()))
            lines.append(' '.join(str(int(v)) for v in valid.ravel()))
        else:
            lines.append(f'op query {arguments[0]:.17g}')
        try:
            if kind == 'push':
                outcome = f'result {int(runtime.push(*arguments))}'
            else:
                pose, calibration = runtime.query_details(*arguments)
                outcome = 'result ok ' + ' '.join(f'{v:.17g}' for v in (*pose.ravel(), *calibration.ravel()))
            if expected is not None:
                raise AssertionError(f'{kind} {arguments[0]} should fail ({expected})')
        except ValueError:
            if expected is None:
                raise
            outcome = f'result error_{expected}'
        lines.append(f'calls {len(recorder.calls)}')
        for name, inputs, output in recorder.calls:
            lines.append(f'call {name} {len(inputs)}')
            lines.extend(tensor_line(value) for value in (*inputs, output))
        lines.append(outcome)
    lines.append('end')
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return sum(1 for kind, _, _ in operations if kind == 'query')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(Path(__file__).resolve().parents[1]/'tests'/'data'))
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        shutil.rmtree(output)
    torch.manual_seed(0)
    # The outlier ray test (off by default) is on, so its paths are checked too.
    model = HandLite(LiteConfig(dim=16, heads=2, blocks=1, dropout=0., slots_per_camera=4,
                                ray_outlier_ratio=6.)).eval()
    for head in (model.decoder.output, model.decoder.gap, model.calibrator.output):   # nonzero: outputs matter
        for parameter in head[-1].parameters():
            torch.nn.init.normal_(parameter, std=.05)
    export_lite(model, output)
    runtime = LiteRuntime(output, backend='onnxruntime')
    queries = write(runtime, scenario(), output/'golden.txt')
    for path in output.glob('*.onnx'):                             # C++ uses the ncnn graphs
        path.unlink()
    meta = json.loads((output/'lite.json').read_text(encoding='utf-8'))
    meta['formats'] = ['ncnn']
    (output/'lite.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'Wrote {output} ({queries} queries)')


if __name__ == '__main__':
    main()
