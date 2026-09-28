"""Evaluate a checkpoint (HandLiteV3 or HandDirect) on fixed validation windows."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from hand_tracking.accelerator import xla_device
from hand_tracking.checkpoints import load_checkpoint
from hand_tracking.data import HandWindows
from hand_tracking.engine import run_epoch


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=['val', 'test'], default='val', help='Use test only after model selection')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--windows-per-clip', type=int, default=16, help='0 evaluates all validation windows')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--device', choices=['cpu', 'cuda', 'xla'], default='cpu')
    p.add_argument('--max-clips', type=int, default=0, help='Development subset; 0 uses all clips')
    p.add_argument('--max-events', type=int, help='Override the checkpoint input layout')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(4)
    device = xla_device() if args.device == 'xla' else torch.device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    config, sampling = checkpoint.model.config, checkpoint.sampling
    if args.max_events is not None:
        sampling = replace(sampling, max_events=args.max_events)
    dataset = HandWindows(args.data, args.split, args.windows_per_clip, max_clips=args.max_clips,
                          context_s=config.context_s, max_events=sampling.max_events, target_frame=sampling.target_frame,
                          mask_out_of_view=sampling.mask_out_of_view)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    options = checkpoint.saved['training_options']
    report = run_epoch(checkpoint.model, loader, device, bone_weight=options['bone_weight'],
                       relative_weight=options.get('relative_weight', 0.), error_weight=options.get('error_weight', 0.),
                       presence_weight=options.get('presence_weight', 0.))
    report.update(checkpoint=str(Path(args.checkpoint).resolve()), split=args.split, windows_per_clip=args.windows_per_clip,
                  max_events=sampling.max_events, target_frame=sampling.target_frame,
                  evaluated_clips=len(dataset.rows), limited_clips=bool(args.max_clips))
    with Path(args.output).open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
