"""Evaluate a checkpoint, or the model-free triangulation anchor baseline, on fixed validation windows."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from hand_tracking.checkpoints import load_checkpoint
from hand_tracking.config import ModelConfig, SamplingConfig
from hand_tracking.data import HandWindows
from hand_tracking.engine import evaluate_anchor, run_epoch


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', help='Trained model; omit with --triangulation-only')
    p.add_argument('--triangulation-only', action='store_true', help='Evaluate the model-free anchor baseline')
    p.add_argument('--calibration-steps', type=int, default=0,
                   help='Baseline only: per-window Gauss-Newton steps; 0 = nominal')
    p.add_argument('--anchor-lookback-s', type=float, default=ModelConfig.anchor_lookback_s,
                   help='Baseline only; 0 holds the last point')
    p.add_argument('--data', default='exports/gigahands_balanced_5gb')
    p.add_argument('--output', required=True)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--windows-per-clip', type=int, default=16, help='0 evaluates all validation windows')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--max-clips', type=int, default=0, help='Development subset; 0 uses all clips')
    p.add_argument('--max-events', type=int, help='Override checkpoint input layout; baseline default is 128')
    args = p.parse_args(argv)
    if bool(args.checkpoint) == args.triangulation_only:
        p.error('Give exactly one of --checkpoint or --triangulation-only')
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if args.triangulation_only:
        # Same window/history layout as the default model.
        checkpoint = None
        config = replace(ModelConfig(), anchor_lookback_s=args.anchor_lookback_s)
        sampling = SamplingConfig()
    else:
        checkpoint = load_checkpoint(args.checkpoint, device)
        config, sampling = checkpoint.model.config, checkpoint.sampling
    if args.max_events is not None:
        sampling = SamplingConfig(max_events=args.max_events)
    dataset = HandWindows(args.data, 'val', args.windows_per_clip, max_clips=args.max_clips,
                          context_s=config.context_s, max_events=sampling.max_events)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    if checkpoint is None:
        report = evaluate_anchor(loader, device, config, args.calibration_steps)
        report.update(method='triangulation_anchor', calibration_steps=args.calibration_steps,
                      anchor_lookback_s=args.anchor_lookback_s)
    else:
        options = checkpoint.saved['training_options']
        report = run_epoch(checkpoint.model, loader, device, bone_weight=options['bone_weight'],
                           calibration_weight=options.get('calibration_weight', 0.))
        report.update(method='transformer', checkpoint=str(Path(args.checkpoint).resolve()))
    report.update(windows_per_clip=args.windows_per_clip, max_events=sampling.max_events,
                  evaluated_clips=len(dataset.rows), limited_clips=bool(args.max_clips))
    with Path(args.output).open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
