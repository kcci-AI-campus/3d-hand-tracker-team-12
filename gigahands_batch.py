"""Stream TAR once after indexing; generate three pose variants per clip."""
import argparse
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tarfile
import time

from gigahands_sim import Config, load_motion, simulate, save_dataset
from motion_archive import list_motion_members, MAX_MEMBER_BYTES
from dataset_quality import provenance, atomic_json


def run(archive, output, cfg, variants=3, max_bytes=5_000_000_000, notify=None, cancelled=lambda: False, resume=False):
    archive, output = Path(archive), Path(output)
    output.mkdir(parents=True, exist_ok=resume)
    identity=provenance(archive,cfg)
    if resume:
        if json.loads((output/'provenance.json').read_text())!=identity: raise ValueError('Resume provenance mismatch')
    else: atomic_json(output/'provenance.json',identity)
    started = time.monotonic()
    state = dict(status='indexing', archive=str(archive.resolve()), clips_total=0,
                 clips_done=0, clips_failed=0, files=0, bytes=0, limit_bytes=max_bytes,
                 variants=variants, current='', elapsed_s=0)
    saved = {}
    completed = set()
    if resume:
        if json.loads((output/'config.json').read_text(encoding='utf-8')) != asdict(cfg):
            raise ValueError('Resume configuration differs from original')
        for line in (output/'manifest.jsonl').read_text(encoding='utf-8').splitlines():
            record = json.loads(line)
            if 'file' not in record: continue
            if (output/record['file']).stat().st_size != record['bytes']:
                raise ValueError('Existing output size mismatch')
            saved[record['source'], record['variant']] = record
        completed = {source for source, _ in saved if all((source, v) in saved for v in range(1, variants+1))}
        state.update(clips_done=len(completed), files=len(saved), bytes=sum(r['bytes'] for r in saved.values()))
    def report():
        state['elapsed_s'] = round(time.monotonic()-started, 1)
        temporary = output/'progress.tmp'
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        # Windows readers can briefly hold the destination without delete sharing.
        for attempt in range(20):
            try:
                temporary.replace(output/'progress.json')
                break
            except PermissionError:
                time.sleep(.1)
        else:
            print('Progress snapshot busy; continuing with log and manifest.', flush=True)
        print(f"{state['status']} | clips {state['clips_done']}/{state['clips_total']} | "
              f"files {state['files']} | {state['bytes']/1e9:.3f}/{max_bytes/1e9:g} GB | "
              f"{state['elapsed_s']:.0f}s | {state['current']}", flush=True)
        if notify: notify(dict(state))
    (output/'config.json').write_text(json.dumps(asdict(cfg), indent=2), encoding='utf-8')
    report()
    try:
        members = list_motion_members(archive)
        wanted = set(members)
        state.update(clips_total=len(members), status='running')
        report()
        with (output/'manifest.jsonl').open('a' if resume else 'w', encoding='utf-8') as manifest, tarfile.open(archive, 'r|*') as tar:
            clip_index = -1
            for info in tar:
                if info.name not in wanted: continue
                clip_index += 1
                if info.name in completed: continue
                if state['bytes'] >= max_bytes and not any(source == info.name for source, _ in saved):
                    state['status'] = 'size_limit'; break
                if cancelled():
                    state['status'] = 'cancelled'; break
                state['current'] = info.name
                try:
                    if not info.isfile() or not 0 < info.size <= MAX_MEMBER_BYTES:
                        raise ValueError('Invalid archive member')
                    with tar.extractfile(info) as source:
                        raw = source.read(MAX_MEMBER_BYTES+1)
                    if len(raw) != info.size: raise ValueError('Truncated member')
                    points, times = load_motion(info.name, cfg.source_fps, stream=io.BytesIO(raw))
                    del raw
                except (ValueError, KeyError, OSError) as exc:
                    state['clips_failed'] += 1
                    with (output/'failures.jsonl').open('a',encoding='utf-8') as errors:
                        errors.write(json.dumps(dict(source=info.name, error=str(exc)), ensure_ascii=False)+'\n')
                    manifest.flush(); report(); continue
                # Finish all variants for this clip before observing the size limit.
                for variant in range(variants):
                    if (info.name, variant+1) in saved: continue
                    clip_cfg = replace(cfg, seed=cfg.seed+clip_index,
                                       pose_seed=(cfg.pose_seed if cfg.pose_seed is not None else cfg.seed)+clip_index*variants+variant)
                    try: result = simulate(points, times, clip_cfg)
                    except (ValueError,KeyError) as exc:
                        state['clips_failed']+=1
                        with (output/'failures.jsonl').open('a',encoding='utf-8') as errors:
                            errors.write(json.dumps(dict(source=info.name,variant=variant+1,error=str(exc)))+'\n')
                        continue
                    name = f'clip_{clip_index:05d}_pose_{variant+1:02d}.npz'
                    target = output/name
                    temporary = output/(name+'.part')
                    save_dataset(temporary, result, f'{archive}::{info.name}')
                    size = temporary.stat().st_size
                    temporary.rename(target)
                    record = dict(file=name, source=info.name, clip_index=clip_index, variant=variant+1,
                                  seed=clip_cfg.seed, pose_seed=clip_cfg.pose_seed, bytes=size,
                                  frames=len(result['query_time']), windows=len(result['window_starts']))
                    manifest.write(json.dumps(record, ensure_ascii=False)+'\n'); manifest.flush()
                    state['files'] += 1; state['bytes'] += size
                    del result
                    report()
                state['clips_done'] += 1
                if state['bytes'] >= max_bytes:
                    state['status'] = 'size_limit'; break
            else:
                state['status'] = 'completed'
    except KeyboardInterrupt:
        state['status'] = 'interrupted'
    except Exception as exc:
        state.update(status='failed', error=str(exc))
        raise
    finally:
        report()
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config')
    parser.add_argument('--variants', type=int, default=3)
    parser.add_argument('--max-gb', type=float, default=5)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.variants < 1 or not 0 < args.max_gb < float('inf'):
        parser.error('variants and max-gb must be positive')
    cfg = Config(**json.loads(Path(args.config).read_text(encoding='utf-8'))) if args.config else Config()
    cfg.validate()
    notify = None; stopped = [False]; root = None
    if args.gui:
        import tkinter as tk
        from tkinter import ttk
        root = tk.Tk(); root.title('GigaHands Batch'); root.geometry('650x260')
        label = tk.StringVar(value='Reading archive index...')
        ttk.Label(root, textvariable=label, padding=16, wraplength=610).pack(fill='x')
        bar = ttk.Progressbar(root, maximum=100); bar.pack(fill='x', padx=16, pady=8)
        def stop(): stopped[0] = True; button.configure(text='Stopping after current clip...', state='disabled')
        button = ttk.Button(root, text='Stop', command=stop); button.pack(pady=8)
        root.protocol('WM_DELETE_WINDOW', stop)
        def notify(state):
            label.set(f"{state['status']} | {state['clips_done']:,} / {state['clips_total']:,} clips\n"
                      f"{state['files']:,} files | {state['bytes']/1e9:.3f} / {state['limit_bytes']/1e9:g} GB\n"
                      f"Elapsed: {state['elapsed_s']:.0f}s\n{state['current']}")
            bar['value'] = max(100*state['clips_done']/max(1,state['clips_total']), min(100,100*state['bytes']/state['limit_bytes']))
            root.update()
    state = run(args.input, args.output, cfg, args.variants, int(args.max_gb*1e9), notify, lambda: stopped[0], args.resume)
    if root:
        button.configure(text='Close', state='normal', command=root.destroy)
        root.protocol('WM_DELETE_WINDOW', root.destroy)
        root.after(30000, root.destroy)
        root.mainloop()
    if state['status'] == 'failed': raise SystemExit(1)


if __name__ == '__main__': main()
