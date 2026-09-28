"""Epoch runners shared by training, validation and the model-free anchor baseline."""
import time
import torch
from .constants import NUM_CAMERAS, CALIBRATION_PARAMS
from .contracts import model_inputs
from .data import trim_padding
from .objectives import PoseTotals, WeightedMean, pose_loss, calibration_loss


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def batch_loss(model, batch, output, bone_weight, calibration_weight, world_weight=0.):
    """Pose loss over every query time (target frame, plus world_weight times the world-frame
    loss, which keeps a weak pull toward true world coordinates) plus the weighted
    calibration auxiliary loss.

    Returns (loss, calibration loss, number of known camera-event targets). The simulator's
    true camera correction supervises every accepted event's calibration (never an input);
    padding, dropped captures and real captures (NaN targets) carry none.
    """
    config = model.config
    loss = pose_loss(output.pose, batch['target'], batch['target_mask'], batch['world_unit_cm'], bone_weight)
    if world_weight:
        loss = loss+world_weight*pose_loss(output.pose, batch['target_world'], batch['target_mask'],
                                           batch['world_unit_cm'], bone_weight)
    if output.calibration is None:                                     # a model without camera correction
        return loss, loss.new_zeros(()), loss.new_zeros((), dtype=torch.long)
    target = torch.where(output.accepted[:,:,None,None], batch['calibration_target'][:,None], float('nan'))
    target = target.reshape(-1, NUM_CAMERAS, CALIBRATION_PARAMS)
    auxiliary = calibration_loss(output.calibration.reshape(-1, NUM_CAMERAS, CALIBRATION_PARAMS), target,
                                 config.calibration_rotation_deg, config.calibration_position)
    if calibration_weight and config.calibration_head:
        loss = loss+calibration_weight*auxiliary
    return loss, auxiliary, torch.isfinite(target).all(-1).sum()


# Mixed precision: consecutive overflowing steps GradScaler may skip before training stops.
MAX_SKIPPED_STEPS = 50


def optimizer_step(model, optimizer, scaler, loss):
    """Backward, clip to norm 1 and step; returns whether the gradients were finite.

    With mixed precision a scaled fp16 gradient overflows now and then: GradScaler then
    skips the step and lowers its scale, so a non-finite norm is not an error there.
    In full precision it is (a NaN from the model), so it raises."""
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=not scaler.is_enabled())
    scaler.step(optimizer)
    scaler.update()
    return bool(torch.isfinite(norm))


def run_epoch(model, loader, device, optimizer=None, scaler=None, limit=0, bone_weight=.1, log_every=50,
              calibration_weight=0., world_weight=0.):
    """One pass over loader; trains when an optimizer is given. Metrics use the final query."""
    training = optimizer is not None
    model.train(training)
    pose, loss_mean, calibration = PoseTotals(), WeightedMean(), WeightedMean()
    started, batches, skipped, consecutive = time.monotonic(), 0, 0, 0
    for batch in loader:
        batch = to_device(trim_padding(batch), device)
        with torch.set_grad_enabled(training), \
                torch.autocast(device_type=device.type, enabled=training and device.type == 'cuda'):
            output = model(*model_inputs(batch), return_details=True)
            loss, auxiliary, known = batch_loss(model, batch, output, bone_weight, calibration_weight, world_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        if training:
            if optimizer_step(model, optimizer, scaler, loss):
                consecutive = 0
            else:
                skipped, consecutive = skipped+1, consecutive+1
                if consecutive >= MAX_SKIPPED_STEPS:
                    raise FloatingPointError(f'Non-finite gradients in {consecutive} consecutive steps '
                                             f'(grad scale {scaler.get_scale():.3g}): not an fp16 overflow')
        pose.add(output.pose, batch)
        loss_mean.add(loss.item(), len(output.pose))
        # Match calibration_loss: one item per known camera and event.
        calibration.add(auxiliary.item(), known.item())
        batches += 1
        if batches % log_every == 0:
            print(f"{'train' if training else 'val'} batch={batches} samples={pose.samples} "
                  f"MPJPE={pose.mpjpe_mm:.2f}mm" + (f' skipped_steps={skipped}' if skipped else ''), flush=True)
        if limit and batches >= limit:
            break
    summary = pose.summary()
    return dict(loss=loss_mean.mean, mpjpe_mm=summary['mpjpe_mm'], pck20=summary['pck20'],
                mpjpe_world_mm=summary['mpjpe_world_mm'],
                calibration_mse=calibration.mean, samples=summary['samples'], joints=summary['joints'],
                skipped_steps=skipped,
                batches=batches, seconds=time.monotonic()-started)


def evaluate_anchor(loader, device, config):
    """Final-query metrics of the model-free anchor: HandLite with config's geometry, no
    calibration head and its zero-initialised residual head, so the pose is the anchor."""
    from dataclasses import replace
    from .lite import HandLite
    model = HandLite(replace(config, calibration_head=False)).to(device).eval()
    pose = PoseTotals()
    with torch.inference_mode():
        for batch in loader:
            batch = to_device(trim_padding(batch), device)
            pose.add(model(*model_inputs(batch)), batch)
    return pose.summary()
