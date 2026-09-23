"""Train the GigaHands time-based (camera event) pose model: HandTransformer (default) or
the fixed-size, export-friendly HandLite (--arch lite)."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from hand_tracking.checkpoints import build_model, restore_training, save_checkpoint, training_checkpoint
from hand_tracking.config import ARCHITECTURES, SamplingConfig, add_model_arguments, model_config_from_args
from hand_tracking.data import HandWindows
from hand_tracking.engine import run_epoch

# Settings that must be > 0 and >= 0; the model's own fields are checked by ModelConfig.
POSITIVE = ('epochs', 'batch_size', 'threads', 'lr')
NONNEGATIVE = ('workers', 'seed', 'max_clips', 'max_train_batches', 'max_val_batches', 'bone_weight',
               'weight_decay', 'rig_rotate_deg', 'rig_shift', 'calibration_weight')


def build_parser(argv=None):
    """The model flags depend on --arch, so it is read first."""
    first = argparse.ArgumentParser(add_help=False)
    first.add_argument('--arch', choices=sorted(ARCHITECTURES), default='transformer')
    architecture = first.parse_known_args(argv)[0].arch
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arch', choices=sorted(ARCHITECTURES), default='transformer',
                   help='transformer: HandTransformer; lite: HandLite (ONNX/ncnn deployment)')
    p.add_argument('--data', default='exports/gigahands_balanced_5gb')
    p.add_argument('--output', default='runs/hand_transformer')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=16)
    add_model_arguments(p, ARCHITECTURES[architecture])
    p.add_argument('--max-events', type=int, default=SamplingConfig.max_events,
                   help='Event slots per window (oldest dropped beyond)')
    p.add_argument('--calibration-weight', type=float, default=.01, help='Auxiliary loss on the simulator camera error')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--bone-weight', type=float, default=.1)
    p.add_argument('--rig-rotate-deg', type=float, default=45., help='Train-only random rig+hand rotation; 0 disables')
    p.add_argument('--rig-shift', type=float, default=.25, help='Train-only random translation per axis (world units)')
    p.add_argument('--windows-per-clip', type=int, default=16)
    p.add_argument('--val-windows-per-clip', type=int, default=16)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--max-clips', type=int, default=0, help='Development subset per split; 0 means all')
    p.add_argument('--max-train-batches', type=int, default=0)
    p.add_argument('--max-val-batches', type=int, default=0)
    p.add_argument('--resume', action='store_true', help='Resume output/last.pt with identical settings')
    return p


def parse_settings(parser, argv=None):
    """Arguments plus validated model and sampling configs; errors exit through the parser."""
    args = parser.parse_args(argv)
    if min(getattr(args, name) for name in POSITIVE) <= 0 or not math.isfinite(args.lr):
        parser.error('Invalid training settings')
    if min(getattr(args, name) for name in NONNEGATIVE) < 0:
        parser.error('Settings must be nonnegative')
    try:
        config = model_config_from_args(args, ARCHITECTURES[args.arch])
        return args, config, SamplingConfig(max_events=args.max_events)
    except ValueError as error:
        parser.error(str(error))


def select_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this PyTorch installation')
    return torch.device(name)


def record_run(parser, args, output):
    """Write run.json, or on resume check it matches; returns the recorded options."""
    options = vars(args).copy()
    options.pop('resume')
    options['manifest_sha256'] = hashlib.sha256((Path(args.data)/'manifest.jsonl').read_bytes()).hexdigest()
    path = output/'run.json'
    if args.resume:
        previous = json.loads(path.read_text())
        # Flags added after that run was started match when they are at their defaults.
        previous.update({k: v for k, v in options.items() if k not in previous and v == parser.get_default(k)})
        if previous != options:
            raise ValueError('Resume settings/dataset differ')
    else:
        path.write_text(json.dumps(options, indent=2), encoding='utf-8')
    return options


def build_loaders(args, config, sampling, device):
    train = HandWindows(args.data, 'train', windows_per_clip=args.windows_per_clip, seed=args.seed,
                        max_clips=args.max_clips, rig_rotate_deg=args.rig_rotate_deg, rig_shift=args.rig_shift,
                        context_s=config.context_s, max_events=sampling.max_events)
    val = HandWindows(args.data, 'val', windows_per_clip=args.val_windows_per_clip, seed=args.seed,
                      max_clips=args.max_clips, context_s=config.context_s, max_events=sampling.max_events)
    return [DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                       pin_memory=device.type == 'cuda') for dataset in (train, val)]


def main(argv=None):
    parser = build_parser(argv)
    args, config, sampling = parse_settings(parser, argv)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=args.resume)
    options = record_run(parser, args, output)
    train_loader, val_loader = build_loaders(args, config, sampling, device)
    model = build_model(args.arch, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    first, best = 0, float('inf')
    if args.resume:
        first, best = restore_training(output/'last.pt', model, optimizer, scheduler, scaler)
    print(f'device={device} parameters={sum(p.numel() for p in model.parameters()):,} '
          f'train_windows/epoch<={len(train_loader.dataset):,} val_windows<={len(val_loader.dataset):,} '
          f'(events from {config.context_s:.2f} s before the first query)', flush=True)
    if args.max_clips or args.max_train_batches or args.max_val_batches:
        print('DEVELOPMENT RUN: metrics are not full-dataset performance.', flush=True)
    losses = dict(bone_weight=args.bone_weight, calibration_weight=args.calibration_weight)
    for epoch in range(first, args.epochs):
        train_loader.dataset.epoch = epoch
        train = run_epoch(model, train_loader, device, optimizer, scaler, args.max_train_batches, **losses)
        val = run_epoch(model, val_loader, device, limit=args.max_val_batches, **losses)
        scheduler.step()
        improved = val['mpjpe_mm'] < best
        best = min(best, val['mpjpe_mm'])
        with (output/'metrics.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(epoch=epoch+1, train=train, val=val))+'\n')
        saved = training_checkpoint(model, sampling, optimizer, scheduler, scaler, epoch, best, options)
        save_checkpoint(output/'last.pt', saved)
        if improved:
            save_checkpoint(output/'best.pt', saved)
        print(f"epoch={epoch+1}/{args.epochs} train={train['mpjpe_mm']:.2f}mm val={val['mpjpe_mm']:.2f}mm "
              f"PCK20={val['pck20']:.3f}", flush=True)


if __name__ == '__main__':
    main()
