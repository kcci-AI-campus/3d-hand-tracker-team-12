"""Train a pose model on GigaHands camera-event windows: HandLiteV3 (default) or the
networks-only comparison HandDirect (--arch direct)."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from hand_tracking.accelerator import is_xla, manual_seed, xla_device
from hand_tracking.checkpoints import build_model, restore_training, save_checkpoint, training_checkpoint
from hand_tracking.config import (ARCHITECTURES, DEFAULT_ARCHITECTURE, SamplingConfig, add_model_arguments,
                                  model_config_from_args)
from hand_tracking.data import HandWindows
from hand_tracking.engine import run_epoch

# Settings that must be > 0 and >= 0; the model's own fields are checked by its config.
POSITIVE = ('epochs', 'batch_size', 'threads', 'lr', 'prefetch', 'reprojection_queries')
NONNEGATIVE = ('workers', 'seed', 'max_clips', 'max_train_batches', 'max_val_batches', 'bone_weight', 'train_queries',
               'weight_decay', 'relative_weight', 'error_weight', 'presence_weight', 'stage_weight',
               'reprojection_weight')
# Settings that change speed, not results: a resume may differ in them.
RUNTIME = ('workers', 'threads', 'cache_dir', 'prefetch')


def build_parser(argv=None):
    """The model flags depend on --arch, so it is read first."""
    first = argparse.ArgumentParser(add_help=False)
    first.add_argument('--arch', choices=sorted(ARCHITECTURES), default=DEFAULT_ARCHITECTURE)
    architecture = first.parse_known_args(argv)[0].arch
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arch', choices=sorted(ARCHITECTURES), default=DEFAULT_ARCHITECTURE,
                   help='litev3: HandLiteV3; direct: HandDirect (networks only, comparison)')
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=85)
    p.add_argument('--batch-size', type=int, default=128)
    add_model_arguments(p, ARCHITECTURES[architecture])
    p.add_argument('--max-events', type=int, default=SamplingConfig.max_events,
                   help='Event slots per window (oldest dropped beyond)')
    p.add_argument('--target-frame', choices=('rig', 'world'), default='rig',
                   help="Pose targets in the placed rig's frame (rig-wide error removed) or the true world")
    p.add_argument('--keep-out-of-view', action='store_true',
                   help='Keep position targets of hands outside all cameras (default: only the in-view loss teaches them)')
    p.add_argument('--relative-weight', type=float, default=1.,
                   help='Weight of the wrist-relative pose loss (hand shape, free of the wrist position error)')
    p.add_argument('--error-weight', type=float, default=.1,
                   help='Weight of the joint error estimate loss (never changes the pose)')
    p.add_argument('--presence-weight', type=float, default=.1,
                   help='Weight of the hand in-view loss (never changes the pose)')
    p.add_argument('--stage-weight', type=float, default=.5,
                   help="Weight of each earlier coarse-to-fine decoder stage's pose loss (HandDirect with refine)")
    p.add_argument('--reprojection-weight', type=float, default=0.,
                   help='Weight of the reprojection loss: the pose at recent frames\' capture times must lie on their '
                        'detected 2D rays (0: off; one extra forward pass per batch)')
    p.add_argument('--reprojection-queries', type=int, default=4,
                   help='Newest detecting frames per window checked by the reprojection loss')
    p.add_argument('--lr', type=float, default=8.5e-4)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--bone-weight', type=float, default=.1)
    p.add_argument('--windows-per-clip', type=int, default=16)
    p.add_argument('--train-queries', type=int, default=4,
                   help="Train on each window's last N queries only (0: all); validation keeps all")
    p.add_argument('--val-windows-per-clip', type=int, default=16)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--prefetch', type=int, default=4, help='Batches each data worker prepares ahead')
    p.add_argument('--cache-dir', default='',
                   help='Folder for uncompressed preprocessed clips, written on first read (about 2x the data size)')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'xla'], default='auto',
                   help='xla: TPU through PyTorch/XLA (float32, fixed batch shapes)')
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
        return args, model_config_from_args(args, ARCHITECTURES[args.arch]), SamplingConfig(max_events=args.max_events, target_frame=args.target_frame,
                                                                    mask_out_of_view=not args.keep_out_of_view)
    except ValueError as error:
        parser.error(str(error))


def select_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this PyTorch installation')
    if name == 'xla':
        return xla_device()
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
        if ({k: v for k, v in previous.items() if k not in RUNTIME}
                != {k: v for k, v in options.items() if k not in RUNTIME}):
            raise ValueError('Resume settings/dataset differ')
    else:
        path.write_text(json.dumps(options, indent=2), encoding='utf-8')
    return options


def build_loaders(args, config, sampling, device):
    train = HandWindows(args.data, 'train', windows_per_clip=args.windows_per_clip, seed=args.seed,
                        max_clips=args.max_clips,
                        context_s=config.context_s, max_events=sampling.max_events, target_frame=sampling.target_frame,
                        queries=args.train_queries, cache=args.cache_dir or None, mask_out_of_view=sampling.mask_out_of_view)
    val = HandWindows(args.data, 'val', windows_per_clip=args.val_windows_per_clip, seed=args.seed,
                      max_clips=args.max_clips, context_s=config.context_s, max_events=sampling.max_events,
                      target_frame=sampling.target_frame, cache=args.cache_dir or None,
                      mask_out_of_view=sampling.mask_out_of_view)
    # XLA compiles once per batch shape: training drops the last, smaller batch of an epoch
    # (validation keeps it, one extra shape compiled once and reused). With many workers each
    # keeps one thread (no oversubscription of the cores) and prefetches several batches.
    # Workers are not persistent: each epoch's workers see the dataset's new epoch (shuffle).
    extra = dict(worker_init_fn=_single_thread, prefetch_factor=args.prefetch) if args.workers else {}
    return [DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                       pin_memory=device.type == 'cuda', drop_last=is_xla(device) and dataset is train, **extra)
            for dataset in (train, val)]


def _single_thread(worker_id):
    """DataLoader worker: one PyTorch thread (numpy's BLAS threads are set by the parent's
    OMP_NUM_THREADS/OPENBLAS_NUM_THREADS)."""
    torch.set_num_threads(1)


def main(argv=None):
    parser = build_parser(argv)
    args, config, sampling = parse_settings(parser, argv)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    manual_seed(device, args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=args.resume)
    options = record_run(parser, args, output)
    train_loader, val_loader = build_loaders(args, config, sampling, device)
    model = build_model(config).to(device)
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
    losses = dict(bone_weight=args.bone_weight, relative_weight=args.relative_weight, error_weight=args.error_weight,
                  presence_weight=args.presence_weight, stage_weight=args.stage_weight,
                  reprojection_weight=args.reprojection_weight, reprojection_queries=args.reprojection_queries)
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
        error_miss = '' if val['error_miss_mm'] is None else f" error_miss={val['error_miss_mm']:.2f}mm"
        if val['reprojection_mrad'] is not None:
            error_miss += f" reproj={val['reprojection_mrad']:.1f}mrad"
        if val['coarse_mpjpe_mm'] is not None:
            error_miss += f" coarse={val['coarse_mpjpe_mm']:.2f}mm"
        if val['presence_accuracy'] is not None:
            error_miss += f" in_view_acc={val['presence_accuracy']*100:.1f}% (out {val['out_of_view_rate']*100:.1f}%)"
        print(f"epoch={epoch+1}/{args.epochs} train={train['mpjpe_mm']:.2f}mm val={val['mpjpe_mm']:.2f}mm "
              f"PCK20={val['pck20']:.3f} rel={val['mpjpe_rel_mm']:.2f}mm{error_miss}", flush=True)
        # Per anchor kind: share of the joints with a target, their MPJPE and (anchored kinds) the
        # anchor's alone on the same joints (final query)
        def kind_text(name):
            text = f"{name}={val['kind_rate'][name]*100:.1f}%"
            if val['kind_mpjpe_mm'][name] is not None:
                text += f"/{val['kind_mpjpe_mm'][name]:.1f}mm"
            alone = val['kind_anchor_mpjpe_mm'].get(name)
            return text if alone is None else text+f"(anchor {alone:.1f}mm)"
        if val['kind_rate']['triangulated'] is not None:        # models with an anchor
            print('  kinds '+' '.join(kind_text(name) for name in val['kind_rate']), flush=True)


if __name__ == '__main__':
    main()
