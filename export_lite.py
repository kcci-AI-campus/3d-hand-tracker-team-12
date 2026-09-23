"""Export a trained HandLite checkpoint for deployment (ONNX and/or ncnn) and check the
torch-free LiteRuntime against the PyTorch LiteStream on a real clip.
Requires requirements-export.txt (ncnn: also `ncnn` and `pnnx`)."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from hand_tracking.checkpoints import load_checkpoint
from hand_tracking.data import read_clip
from hand_tracking.lite import HandLite, LiteStream
from hand_tracking.lite_export import FORMATS, export_lite
from hand_tracking.lite_runtime import LiteRuntime

BACKENDS = {'onnx': 'onnxruntime', 'ncnn': 'ncnn'}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, help='A checkpoint trained with --arch lite')
    p.add_argument('--output', required=True, help='New directory for the graphs and lite.json')
    p.add_argument('--formats', nargs='+', choices=FORMATS, default=list(FORMATS))
    p.add_argument('--check-input', help='NPZ clip streamed through both runtimes and LiteStream')
    p.add_argument('--queries', type=int, default=30, help='Query times compared from the clip')
    p.add_argument('--tolerance', type=float, default=2e-3,
                   help='Largest accepted |runtime - PyTorch| in world units (ncnn may use fp16)')
    return p.parse_args(argv)


def compare_on_clip(model, directory, formats, npz, queries):
    """Stream the clip's events through LiteStream and each runtime; largest pose difference."""
    clip = read_clip(npz)
    order = np.argsort(clip['event_arrival'], kind='stable')
    times = clip['query_time'][np.linspace(len(clip['query_time'])//4, len(clip['query_time'])-1, queries).astype(int)]
    runners = {'pytorch': LiteStream(model)}
    runners.update({name: LiteRuntime(directory, backend=BACKENDS[name]) for name in formats})
    worst, pushed = {name: 0. for name in formats}, 0
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
            worst[name] = max(worst[name], float(np.abs(runners[name].query(time)-expected).max()))
    return worst


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(4)
    checkpoint = load_checkpoint(args.checkpoint)
    if not isinstance(checkpoint.model, HandLite):
        raise SystemExit('Not a HandLite checkpoint; use export_onnx.py for HandTransformer')
    meta = export_lite(checkpoint.model, args.output, args.formats)
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
