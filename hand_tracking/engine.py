"""Epoch runners shared by training, validation and the model-free anchor baseline."""
import time
import torch
from .constants import NUM_CAMERAS, CALIBRATION_PARAMS
from .contracts import model_inputs
from .events import event_anchor
from .objectives import PoseTotals, WeightedMean, pose_loss, calibration_loss


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def batch_loss(model, batch, output, bone_weight, calibration_weight):
    """Pose loss over every query time plus the weighted calibration auxiliary loss.

    Returns (loss, calibration loss, number of known camera-event targets). The simulator's
    true camera correction supervises every accepted event's calibration (never an input);
    padding, dropped captures and real captures (NaN targets) carry none.
    """
    config = model.config
    loss = pose_loss(output.pose, batch['target'], batch['target_mask'], batch['world_unit_cm'], bone_weight)
    target = torch.where(output.accepted[:,:,None,None], batch['calibration_target'][:,None], float('nan'))
    target = target.reshape(-1, NUM_CAMERAS, CALIBRATION_PARAMS)
    auxiliary = calibration_loss(output.calibration.reshape(-1, NUM_CAMERAS, CALIBRATION_PARAMS), target,
                                 config.calibration_rotation_deg, config.calibration_position)
    if calibration_weight and config.calibration_head:
        loss = loss+calibration_weight*auxiliary
    return loss, auxiliary, torch.isfinite(target).all(-1).sum()


def optimizer_step(model, optimizer, scaler, loss):
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()


def run_epoch(model, loader, device, optimizer=None, scaler=None, limit=0, bone_weight=.1, log_every=50,
              calibration_weight=0.):
    """One pass over loader; trains when an optimizer is given. Metrics use the final query."""
    training = optimizer is not None
    model.train(training)
    pose, loss_mean, calibration = PoseTotals(), WeightedMean(), WeightedMean()
    started, batches = time.monotonic(), 0
    for batch in loader:
        batch = to_device(batch, device)
        with torch.set_grad_enabled(training), \
                torch.autocast(device_type=device.type, enabled=training and device.type == 'cuda'):
            output = model(*model_inputs(batch), return_details=True)
            loss, auxiliary, known = batch_loss(model, batch, output, bone_weight, calibration_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        if training:
            optimizer_step(model, optimizer, scaler, loss)
        pose.add(output.pose, batch)
        loss_mean.add(loss.item(), len(output.pose))
        # Match calibration_loss: one item per known camera and event.
        calibration.add(auxiliary.item(), known.item())
        batches += 1
        if batches % log_every == 0:
            print(f"{'train' if training else 'val'} batch={batches} samples={pose.samples} "
                  f"MPJPE={pose.mpjpe_mm:.2f}mm", flush=True)
        if limit and batches >= limit:
            break
    summary = pose.summary()
    return dict(loss=loss_mean.mean, mpjpe_mm=summary['mpjpe_mm'], pck20=summary['pck20'],
                calibration_mse=calibration.mean, samples=summary['samples'], joints=summary['joints'],
                batches=batches, seconds=time.monotonic()-started)


def evaluate_anchor(loader, device, config, calibration_steps=0):
    """Final-query metrics of the model-free anchor with config's anchor settings,
    optionally Gauss-Newton self-calibrated per window."""
    pose = PoseTotals()
    with torch.inference_mode():
        for batch in loader:
            batch = to_device(batch, device)
            anchor = event_anchor(*model_inputs(batch), lookback_s=config.anchor_lookback_s,
                                  hold_s=config.anchor_hold_s, sample_max_age_s=config.sample_max_age_s,
                                  calibration_steps=calibration_steps, gate=config.anchor_ray_gate,
                                  motion_fit=config.anchor_motion_fit, rotation_deg=config.calibration_rotation_deg,
                                  position_std=config.calibration_position)
            pose.add(anchor, batch)
    return pose.summary()
