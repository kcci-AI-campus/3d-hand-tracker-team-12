"""Export a trained checkpoint to ONNX (dynamic batch, event and query counts) and check
it against PyTorch with ONNX Runtime. Requires requirements-export.txt."""
import argparse
import json
from pathlib import Path
import torch
from hand_tracking.checkpoints import load_checkpoint
from hand_tracking.contracts import MODEL_INPUT_KEYS
from hand_tracking.data import read_clip, make_sample
from hand_tracking.export import OUTPUT_NAMES, compare_onnx, example_inputs, export_onnx


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True, help='.onnx file to create')
    parser.add_argument('--opset', type=int, help='ONNX opset; default: the exporter default')
    parser.add_argument('--check-input', help='NPZ clip whose last window is also compared (default: synthetic only)')
    parser.add_argument('--tolerance', type=float, default=1e-4, help='Largest accepted |ONNX - PyTorch|')
    return parser.parse_args(argv)


def check_inputs(checkpoint, npz):
    """Synthetic inputs with other sizes than the traced example, plus a real window if given."""
    cases = {'synthetic': example_inputs(batch=3, events=11, queries=4)}
    if npz:
        clip = read_clip(npz)
        sample = make_sample(clip, len(clip['window_starts'])-1, context_s=checkpoint.model.config.context_s,
                             max_events=checkpoint.sampling.max_events)
        cases['npz'] = tuple(sample[key][None] for key in MODEL_INPUT_KEYS)
    return cases


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    checkpoint = load_checkpoint(args.checkpoint)
    export_onnx(checkpoint.model, output, args.opset)
    differences = {name: compare_onnx(output, checkpoint.model, inputs)
                   for name, inputs in check_inputs(checkpoint, args.check_input).items()}
    report = dict(onnx=str(output.resolve()), inputs=list(MODEL_INPUT_KEYS), outputs=list(OUTPUT_NAMES),
                  max_abs_difference=differences, max_events=checkpoint.sampling.max_events,
                  context_s=checkpoint.model.config.context_s)
    print(json.dumps(report, indent=2), flush=True)
    if max(differences.values()) > args.tolerance:
        raise SystemExit(f'ONNX output differs from PyTorch by more than {args.tolerance}')


if __name__ == '__main__':
    main()
