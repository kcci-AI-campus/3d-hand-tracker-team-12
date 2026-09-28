"""Export a trained checkpoint (HandLite or HandDirect) for deployment (ONNX and/or ncnn) and
check the torch-free runtime against the PyTorch stream on a real clip.
Requires requirements-export.txt (ncnn: also `ncnn` and `pnnx`)."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from hand_tracking.checkpoints import architecture_of, load_checkpoint, stream_for
from hand_tracking.data import read_clip
from hand_tracking.direct_export import export_direct
from hand_tracking.direct_runtime import DirectRuntime
from hand_tracking.lite_export import FORMATS, export_lite
from hand_tracking.lite_runtime import LiteRuntime

BACKENDS = {'onnx': 'onnxruntime', 'ncnn': 'ncnn'}
# architecture -> (export function, torch-free runtime)
DEPLOYMENT = {'lite': (export_lite, LiteRuntime), 'direct': (export_direct, DirectRuntime)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, help='A trained checkpoint (any architecture)')
    p.add_argument('--output', required=True, help='New directory for the graphs and their metadata')
    p.add_argument('--formats', nargs='+', choices=FORMATS, default=list(FORMATS))
    p.add_argument('--check-input', help='NPZ clip streamed through each runtime and the PyTorch stream')
    p.add_argument('--queries', type=int, default=30, help='Query times compared from the clip (> 0)')
    p.add_argument('--tolerance', type=float, default=2e-3,
                   help='Largest accepted |runtime - PyTorch| in world units (ncnn may use fp16)')
    args = p.parse_args(argv)
    if args.queries < 1 or not args.tolerance > 0:
        p.error('--queries and --tolerance must be positive')
    return args


def compare_on_clip(model, directory, formats, npz, queries):
    """Stream the clip's events through the PyTorch stream and each runtime; largest pose difference."""
    clip = read_clip(npz)
    order = np.argsort(clip['event_arrival'], kind='stable')
    times = clip['query_time'][np.linspace(len(clip['query_time'])//4, len(clip['query_time'])-1, queries).astype(int)]
    runtime = DEPLOYMENT[architecture_of(model)][1]
    runners = {'pytorch': stream_for(model)}
    runners.update({name: runtime(directory, backend=BACKENDS[name]) for name in formats})
    worst, pushed, compared = {name: 0. for name in formats}, 0, 0
    for time in times:
        while pushed < len(order) and clip['event_arrival'][order[pushed]] <= time:
            i = order[pushed]
            pushed += 1
            for runner in runners.values():
                runner.push(clip['event_camera'][i], clip['event_features'][i], clip['event_valid'][i],
                            clip['event_capture'][i], clip['event_arrival'][i])
        if not pushed:
            continue
        expected = runners['pytorch'].query(time).numpy()
        for name in formats:
            worst[name] = max(worst[name], pose_difference(runners[name].query(time), expected, name, time))
        compared += 1
    if not compared:
        raise ValueError(f'No query of {npz} had an arrived event to compare')
    return worst


def pose_difference(actual, expected, name, time):
    """Largest |actual - expected|; non-finite values fail (max() would silently drop NaN)."""
    if not np.isfinite(expected).all():
        raise FloatingPointError(f'PyTorch pose is not finite at {time}')
    if not np.isfinite(actual).all():
        raise FloatingPointError(f'{name} pose is not finite at {time}')
    return float(np.abs(actual-expected).max())


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(4)
    checkpoint = load_checkpoint(args.checkpoint)
    export = DEPLOYMENT[architecture_of(checkpoint.model)][0]
    meta = export(checkpoint.model, args.output, args.formats)
    report = dict(output=str(Path(args.output).resolve()), formats=args.formats, config=meta['config'],
                  graphs=meta['graphs'])
    if args.check_input:
        report['max_abs_difference'] = compare_on_clip(checkpoint.model, args.output, args.formats, args.check_input,
                                                       args.queries)
    print(json.dumps(report, indent=2), flush=True)
    if args.check_input and max(report['max_abs_difference'].values()) > args.tolerance:
        raise SystemExit(f'Runtime differs from PyTorch by more than {args.tolerance}')


if __name__ == '__main__':
    main()
