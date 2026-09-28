"""Predict one NPZ window from a trained checkpoint."""
import argparse
from pathlib import Path
import numpy as np
import torch
from hand_tracking.checkpoints import load_checkpoint
from hand_tracking.contracts import model_inputs
from hand_tracking.data import read_clip, make_sample


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--window-index', type=int, default=-1)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(4)
    checkpoint = load_checkpoint(args.checkpoint)
    model = checkpoint.model
    clip = read_clip(args.input)
    window = args.window_index % len(clip['window_starts'])
    sample = make_sample(clip, window, context_s=model.config.context_s, max_events=checkpoint.sampling.max_events,
                         target_frame=checkpoint.sampling.target_frame, mask_out_of_view=checkpoint.sampling.mask_out_of_view)
    if not sample['event_present'].any():
        raise ValueError('No arrived observations in this window')
    with torch.inference_mode():
        output = model(*(value[None] for value in model_inputs(sample)), return_details=True)
    if not bool(output.accepted[0, -1]):
        raise ValueError('The model used no event for the final query of this window')
    pose = output.pose[0, -1].numpy()
    # Each joint's anchor and how it was anchored (model.ANCHOR_KINDS; HandLiteV3 only), and its expected error in mm.
    extra = {}
    if output.anchored is not None:
        extra.update(anchor_xyz=output.anchor_xyz[0, -1].numpy(), anchor_kind=output.anchored[0, -1].numpy())
    if output.error is not None:
        extra['expected_error_mm'] = torch.expm1(output.error[0, -1]).numpy()
    if output.presence is not None:                           # each hand: probability of being in some camera's view
        extra['in_view_probability'] = torch.sigmoid(output.presence[0, -1]).numpy()
    extra['hand_in_view'] = sample['hand_in_view'][-1].numpy()
    with Path(args.output).open('xb') as stream:
        np.savez_compressed(stream, predicted_xyz=pose, predicted_xyz_cm=pose*clip['world_unit_cm'],
                            target_xyz=sample['target'][-1].numpy(), target_mask=sample['target_mask'][-1].numpy(),
                            world_unit_cm=clip['world_unit_cm'], window_index=window, **extra)
    print(f'Saved {args.output}: [2,21,3] world coordinates and cm')


if __name__ == '__main__':
    main()
