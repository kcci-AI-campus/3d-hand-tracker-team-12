"""Export a trained checkpoint (HandLiteV3 or HandDirect) to ncnn and check the torch-free runtime
against the PyTorch stream on a real clip. Needs `ncnn` and `pnnx`."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from hand_tracking.checkpoints import load_checkpoint, stream_for
from hand_tracking.config import architecture_of
from hand_tracking.data import read_clip
from hand_tracking.deploy import export_ncnn
from hand_tracking.direct_export import export_direct
from hand_tracking.direct_runtime import DirectRuntime
from hand_tracking.runtime import LiteV3Runtime

# Architecture -> (exporter, runtime of its export directory)
EXPORTERS = {'litev3': (export_ncnn, LiteV3Runtime), 'direct': (export_direct, DirectRuntime)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True, help='New directory for the graphs and their metadata')
    p.add_argument('--check-input', help='NPZ clip streamed through the runtime and the PyTorch stream')
    p.add_argument('--queries', type=int, default=30, help='Query times compared from the clip (> 0)')
    p.add_argument('--tolerance', type=float, default=2e-3,
                   help='Largest accepted |runtime - PyTorch| in world units (ncnn uses fp16)')
    args = p.parse_args(argv)
    if args.queries < 1 or not args.tolerance > 0:
        p.error('--queries and --tolerance must be positive')
    return args


def compare_on_clip(model, directory, npz, queries):
    """Stream the clip's events through the PyTorch stream and the runtime; largest pose difference."""
    clip = read_clip(npz)
    order = np.argsort(clip['event_arrival'], kind='stable')
    times = clip['query_time'][np.linspace(len(clip['query_time'])//4, len(clip['query_time'])-1, queries).astype(int)]
    pytorch, runtime = stream_for(model), EXPORTERS[architecture_of(model.config)][1](directory)
    worst, pushed, compared = 0., 0, 0
    for time in times:
        while pushed < len(order) and clip['event_arrival'][order[pushed]] <= time:
            i = order[pushed]
            pushed += 1
            for runner in (pytorch, runtime):
                runner.push(clip['event_camera'][i], clip['event_features'][i], clip['event_valid'][i],
                            clip['event_capture'][i], clip['event_arrival'][i])
        if not pushed:
            continue
        expected, actual = pytorch.query(time).numpy(), runtime.query(time)
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise FloatingPointError(f'Non-finite pose at {time}')
        worst = max(worst, float(np.abs(actual-expected).max()))
        compared += 1
    if not compared:
        raise ValueError(f'No query of {npz} had an arrived event to compare')
    return worst


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(4)
    checkpoint = load_checkpoint(args.checkpoint)
    meta = EXPORTERS[architecture_of(checkpoint.model.config)][0](checkpoint.model, args.output)
    report = dict(output=str(Path(args.output).resolve()), config=meta['config'], graphs=meta['graphs'])
    if args.check_input:
        report['max_abs_difference'] = compare_on_clip(checkpoint.model, args.output, args.check_input, args.queries)
    print(json.dumps(report, indent=2), flush=True)
    if args.check_input and report['max_abs_difference'] > args.tolerance:
        raise SystemExit(f'Runtime differs from PyTorch by more than {args.tolerance}')


if __name__ == '__main__':
    main()
