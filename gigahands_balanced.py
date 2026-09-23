"""Participant-balanced GigaHands generation with participant-disjoint splits."""
import argparse
from collections import defaultdict, deque
from dataclasses import asdict, replace
import io
import json
from pathlib import Path, PurePosixPath
import queue
import random
import re
import tarfile
import tempfile
import threading
import time

from gigahands_sim import Config, load_motion, simulate, save_dataset
from motion_archive import MAX_MEMBER_BYTES


def balanced_plan(records, seed=42):
    """Round-robin participants and their activities; shuffle clips within activity."""
    rng = random.Random(seed)
    grouped = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[record['participant']][record['activity']].append(record)
    participants = sorted(grouped)
    if len(participants) < 2:
        raise ValueError('At least two participants are needed for disjoint splits')
    shuffled = participants.copy(); rng.shuffle(shuffled)
    val = set(shuffled[:max(1, round(len(participants)*.2))])
    splits = {p: ('val' if p in val else 'train') for p in participants}
    per_person = {}
    for participant in participants:
        activities = sorted(grouped[participant]); rng.shuffle(activities)
        pending = []
        for activity in activities:
            clips = sorted(grouped[participant][activity], key=lambda r: r['source'])
            rng.shuffle(clips); pending.append(deque(clips))
        ordered = deque()
        while pending:
            for clips in pending: ordered.append(clips.popleft())
            pending = [clips for clips in pending if clips]
        per_person[participant] = ordered
    plan = []
    while per_person:
        turn = sorted(per_person); rng.shuffle(turn)
        for participant in turn:
            record = dict(per_person[participant].popleft(), split=splits[participant])
            plan.append(record)
            if not per_person[participant]: del per_person[participant]
    return plan, splits


def run(archive, output, cfg, max_bytes=5_000_000_000, notify=None, cancelled=lambda: False):
    archive, output = Path(archive).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for split in ('train', 'val'): (output/split).mkdir()
    (output/'config.json').write_text(json.dumps(asdict(cfg), indent=2), encoding='utf-8')
    start = time.monotonic()
    state = dict(status='indexing', clips_total=0, clips_done=0, clips_failed=0, bytes=0,
                 limit_bytes=max_bytes, train=0, val=0, participants=0, current='')
    def report():
        state['elapsed_s'] = round(time.monotonic()-start, 1)
        temp = output/'progress.tmp'
        temp.write_text(json.dumps(state, indent=2), encoding='utf-8')
        for _ in range(20):
            try: temp.replace(output/'progress.json'); break
            except PermissionError: time.sleep(.1)
        capacity = f'{max_bytes/1e9:g} GB' if max_bytes is not None else 'all clips'
        print(f"{state['status']} | {state['clips_done']}/{state['clips_total']} clips | "
              f"{state['bytes']/1e9:.3f} GB / {capacity} | train {state['train']} val {state['val']} | "
              f"{state['elapsed_s']:.0f}s | {state['current']}", flush=True)
        if notify: notify(dict(state))
    report()
    try:
        # Numeric offsets only: archive member names never become extracted paths.
        with tempfile.TemporaryFile(dir=output) as cache:
            records = []; seen = set()
            with tarfile.open(archive, 'r|*') as tar:
                for info in tar:
                    if cancelled(): state['status'] = 'cancelled'; return state
                    path = PurePosixPath(info.name)
                    if not info.isfile() or path.suffix.lower() not in ('.json', '.npy', '.npz'): continue
                    if not any(p.startswith('keypoints_3d') for p in path.parts): continue
                    if not 0 < info.size <= MAX_MEMBER_BYTES: continue
                    match = re.match(r'^(p\d+)(?:-|$)', path.parts[0])
                    if not match: raise ValueError(f'Cannot identify participant: {info.name}')
                    if info.name in seen: raise ValueError(f'Duplicate member: {info.name}')
                    seen.add(info.name)
                    offset = cache.tell()
                    with tar.extractfile(info) as stream:
                        raw = stream.read(MAX_MEMBER_BYTES+1)
                    if len(raw) != info.size: raise ValueError('Truncated archive member')
                    cache.write(raw); del raw
                    records.append(dict(source=info.name, participant=match.group(1), activity=path.parts[0], offset=offset, size=info.size))
                    if len(records) % 100 == 0:
                        state.update(clips_total=len(records), current=f'Indexing {len(records)} clips'); report()
            plan, splits = balanced_plan(records, cfg.seed)
            split_info = dict(seed=cfg.seed, method='participant-disjoint, approximately 80/20 participants',
                              participants=splits, original_clips=len(records), variants_per_clip=1)
            (output/'split.json').write_text(json.dumps(split_info, indent=2), encoding='utf-8')
            (output/'selection_plan.json').write_text(json.dumps([{k:v for k,v in r.items() if k not in ('offset','size')} for r in plan], indent=2), encoding='utf-8')
            state.update(status='running', clips_total=len(records), participants=len(splits))
            report()
            with (output/'manifest.jsonl').open('w', encoding='utf-8') as manifest:
                for index, row in enumerate(plan):
                    if cancelled(): state['status'] = 'cancelled'; break
                    state['current'] = row['source']
                    cache.seek(row['offset'])
                    raw = cache.read(row['size'])
                    clip_cfg = replace(cfg, seed=cfg.seed+index, pose_seed=cfg.seed+100000+index)
                    try:
                        p, t = load_motion(row['source'], cfg.source_fps, stream=io.BytesIO(raw))
                        del raw
                        result = simulate(p, t, clip_cfg)
                    except (ValueError, KeyError) as exc:
                        state['clips_failed'] += 1
                        manifest.write(json.dumps(dict(source=row['source'], error=str(exc)))+'\n')
                        manifest.flush(); report(); continue
                    name = f"{row['split']}/clip_{index:05d}.npz"
                    temp = output/(name+'.part')
                    save_dataset(temp, result, f"{archive}::{row['source']}")
                    size = temp.stat().st_size; temp.rename(output/name)
                    record = dict(file=name, source=row['source'], participant=row['participant'], activity=row['activity'],
                                  split=row['split'], bytes=size, seed=clip_cfg.seed, pose_seed=clip_cfg.pose_seed,
                                  frames=len(result['query_time']), windows=len(result['window_starts']))
                    manifest.write(json.dumps(record)+'\n'); manifest.flush()
                    del result
                    state['clips_done'] += 1; state[row['split']] += 1; state['bytes'] += size
                    report()
                    if max_bytes is not None and state['bytes'] >= max_bytes: state['status'] = 'size_limit'; break
                else: state['status'] = 'completed'
    except KeyboardInterrupt:
        state['status'] = 'interrupted'
    except Exception as exc:
        state.update(status='failed', error=str(exc)); raise
    finally:
        report()
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-gb', type=float, default=5)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--all-clips', action='store_true', help='Process every clip without a size limit')
    args = parser.parse_args()
    if not 0 < args.max_gb < float('inf'): parser.error('max-gb must be positive')
    limit = None if args.all_clips else int(args.max_gb*1e9)
    if not args.gui:
        run(args.input, args.output, Config(), limit); return
    import tkinter as tk
    from tkinter import ttk
    root = tk.Tk(); root.title('Balanced Dataset'); root.geometry('680x280')
    text = tk.StringVar(value='Preparing all participants...')
    ttk.Label(root, textvariable=text, padding=16, wraplength=640).pack(fill='x')
    bar = ttk.Progressbar(root, maximum=100); bar.pack(fill='x', padx=16, pady=12)
    stopped = threading.Event(); events = queue.Queue()
    def stop(): stopped.set(); button.configure(text='Stopping...', state='disabled')
    button = ttk.Button(root, text='Stop', command=stop); button.pack(pady=8)
    root.protocol('WM_DELETE_WINDOW', stop)
    def work():
        try: run(args.input, args.output, Config(), limit, lambda s: events.put(s), stopped.is_set)
        except Exception as exc: events.put(dict(status='failed', error=str(exc)))
        finally: events.put(None)
    def poll():
        try:
            while True:
                state = events.get_nowait()
                if state is None:
                    button.configure(text='Close', state='normal', command=root.destroy)
                    root.protocol('WM_DELETE_WINDOW', root.destroy)
                    root.after(30000, root.destroy); return
                text.set(f"{state['status']} | {state.get('clips_done',0):,} / {state.get('clips_total',0):,} clips\n"
                         f"{state.get('bytes',0)/1e9:.3f} GB | Train {state.get('train',0)} / Val {state.get('val',0)}\n"
                         f"Participants: {state.get('participants',0)} | Elapsed {state.get('elapsed_s',0):.0f}s\n"
                         f"{state.get('error',state.get('current',''))}")
                bar['value'] = (100*state.get('clips_done',0)/max(1,state.get('clips_total',0)) if args.all_clips
                                else min(100, state.get('bytes',0)/(args.max_gb*1e9)*100))
        except queue.Empty: pass
        root.after(100, poll)
    threading.Thread(target=work, daemon=True).start(); poll(); root.mainloop()


if __name__ == '__main__': main()
