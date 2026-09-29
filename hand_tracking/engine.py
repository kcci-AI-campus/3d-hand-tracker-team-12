"""Epoch runner shared by training and validation (CPU, CUDA or XLA/TPU; see accelerator)."""
import contextlib
import time
import torch
from .accelerator import device_batches, is_xla, sync
from .constants import RESIDUAL_UNIT
from .contracts import model_inputs
from .data import trim_padding
from .model import ANCHOR_KINDS
from .objectives import PoseTotals, WeightedMean, _number, pose_loss, relative_loss


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def joint_error_log(pose, batch):
    """Each joint's actual error log(1 + mm) [B,Q,2,21] (detached) and where it has a target."""
    mask = batch['target_mask'].bool()
    mm = batch['world_unit_cm'].reshape(-1, *[1]*(pose.ndim-2))*10
    error = (pose.detach().float()-torch.where(mask[..., None], batch['target'], 0.)).norm(dim=-1)*mm
    return torch.log1p(error), mask.float()


def pose_objective(pose, batch, bone_weight, relative_weight):
    """Pose loss of one pose [B,Q,2,21,3] plus relative_weight times its wrist-relative loss."""
    loss = pose_loss(pose, batch['target'], batch['target_mask'], batch['world_unit_cm'], bone_weight)
    if relative_weight:
        loss = loss+relative_weight*relative_loss(pose, batch['target'], batch['target_mask'], batch['world_unit_cm'])
    return loss


def batch_loss(batch, output, bone_weight, relative_weight=0., error_weight=0., presence_weight=0., stage_weight=0.):
    """Pose loss in the target frame (the rig frame: the rig-wide error no observation reveals is
    not a target) over every query time, plus relative_weight times the wrist-relative loss (hand
    shape, free of the wrist position error), plus error_weight times the error estimate loss
    (SmoothL1 of output.error against each joint's actual error, log(1 + mm), over joints with a
    target; the estimate's input is detached, so this never changes the pose), plus
    presence_weight times the in-view loss (binary cross-entropy of output.presence against
    hand_in_view, every hand; also on detached tokens). A hand out of all cameras' view has no
    position targets (data.make_sample) and only teaches that it is out of view. stage_weight
    adds each earlier coarse-to-fine stage's pose objective (HandDirect with refine)."""
    loss = pose_objective(output.pose, batch, bone_weight, relative_weight)
    if stage_weight and output.stages is not None:
        for stage in output.stages:
            loss = loss+stage_weight*pose_objective(stage, batch, bone_weight, relative_weight)
    if error_weight and output.error is not None:
        actual, mask = joint_error_log(output.pose, batch)
        misses = torch.nn.functional.smooth_l1_loss(output.error, actual, beta=.1, reduction='none')
        loss = loss+error_weight*(misses*mask).sum()/mask.sum().clamp_min(1)
    if presence_weight and output.presence is not None:
        loss = loss+presence_weight*torch.nn.functional.binary_cross_entropy_with_logits(
            output.presence, batch['hand_in_view'].float())
    return loss


def reprojection_loss(model, batch, queries):
    """Self-consistency with the cameras' own 2D detections: the model is queried again at the
    capture times of each window's `queries` newest events that detected anything (it cannot use
    an event before it arrives, so it never sees the frame it is checked against), and each
    predicted joint must lie on that event's ray: the angle between the ray (the input's nominal
    u,v ray) and the direction from the ray's origin to the joint, i.e. the reprojection error over
    the focal length (0.01 rad is about 3.8 px at 320x240, 55 degrees). SmoothL1 on the angle in
    radians with beta RESIDUAL_UNIT (0.01 rad): linear beyond it, so robust to false detections, and
    about the pose loss's size (a 2 cm miss at 60 cm is 0.033 rad; pose_loss is in metres).
    Returns (loss, summed angle in rad, detected joints), fixed shapes for XLA."""
    features, valid, camera, capture, arrival, present = model_inputs(batch)[:6]
    b, e = present.shape
    detected = present.bool() & valid.flatten(2).any(-1)                                    # [B,E]
    score = torch.where(detected, capture.float(), torch.full_like(capture.float(), -1e9))
    if e < queries:
        score = torch.nn.functional.pad(score, (0, queries-e), value=-1e9)
    score, index = score.topk(queries, dim=-1)                                              # newest captures
    chosen = score > -1e8                                                                   # [B,K]
    index = index.clamp_max(e-1)
    rows = torch.arange(b, device=index.device)[:, None]
    query = torch.where(chosen, score, torch.zeros_like(score))
    pose = model(features, valid, camera, capture, arrival, present, query).float()          # [B,K,2,21,3]
    rays = features[rows, index].float()                                                    # [B,K,2,21,14]
    seen = valid[rows, index].bool() & chosen[..., None, None]                              # [B,K,2,21]
    direction = rays[..., 5:8]/rays[..., 5:8].norm(dim=-1, keepdim=True).clamp_min(1e-8)
    relative = pose-rays[..., 2:5]
    angle = torch.atan2(torch.linalg.cross(relative, direction).norm(dim=-1), (relative*direction).sum(-1))
    angle = torch.where(seen, angle, torch.zeros_like(angle))
    misses = torch.nn.functional.smooth_l1_loss(angle, torch.zeros_like(angle), beta=RESIDUAL_UNIT, reduction='none')
    count = seen.sum()
    return (misses*seen).sum()/count.clamp_min(1), angle.detach().sum(), count


# Mixed precision: consecutive overflowing steps GradScaler may skip before training stops.
MAX_SKIPPED_STEPS = 50


def optimizer_step(model, optimizer, scaler, loss, xla=False):
    """Backward, clip to norm 1 and step; returns whether the gradients were finite (on XLA
    as a device flag, read later: reading it now would end the traced step early).

    With mixed precision a scaled fp16 gradient overflows now and then: GradScaler then
    skips the step and lowers its scale, so a non-finite norm is not an error there.
    In full precision it is (a NaN from the model), so it raises; on XLA (float32, no
    scaler) run_epoch raises at its next read instead."""
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.,
                                          error_if_nonfinite=not scaler.is_enabled() and not xla)
    scaler.step(optimizer)
    scaler.update()
    finite = torch.isfinite(norm)
    return finite if xla else bool(finite)


def _check_finite(nonfinite, phase):
    """XLA: raise when any loss or gradient norm so far was not finite (reads the count)."""
    if int(_number(nonfinite)):
        raise FloatingPointError(f'Nonfinite loss or gradients in {phase} (XLA, float32)')


def run_epoch(model, loader, device, optimizer=None, scaler=None, limit=0, bone_weight=.1, log_every=50,
              relative_weight=0., error_weight=0., presence_weight=0., stage_weight=0., reprojection_weight=0.,
              reprojection_queries=4):
    """One pass over loader; trains when an optimizer is given. Metrics use the final query:
    MPJPE and PCK, the error estimate's miss, and per anchor kind (ANCHOR_KINDS) the share of
    joints with a target, their MPJPE and their anchor's alone (what the correction starts from;
    None for a model without an anchor, HandDirect), and coarse_mpjpe_mm, the first
    coarse-to-fine stage's MPJPE (None for a single-stage model). With reprojection_weight, the
    reprojection loss (reprojection_loss, one extra forward pass per batch) is added and
    reprojection_mrad reports its mean angle (None without it).

    On XLA (TPU) batches keep their padded shape, float32 throughout, each batch ends with
    sync() and the device is read only every log_every batches and at the end (the first
    batch of each shape also compiles; its time is printed)."""
    training = optimizer is not None
    phase = 'train' if training else 'val'
    xla = is_xla(device)
    model.train(training)
    pose, coarse, loss_mean, error_miss = PoseTotals(), PoseTotals(), WeightedMean(), WeightedMean()
    reprojection = WeightedMean()   # mean angle (rad) over the detected joints checked
    presence_hit, out_of_view = WeightedMean(), WeightedMean()   # final query, every hand
    kind_rate = {name: WeightedMean() for name in ANCHOR_KINDS}
    kind_mm = {name: WeightedMean() for name in ANCHOR_KINDS}
    kind_anchor_mm = {name: WeightedMean() for name in ANCHOR_KINDS[1:]}
    started, batches, skipped, consecutive, nonfinite = time.monotonic(), 0, 0, 0, 0
    for batch in device_batches(loader, device):
        batch = batch if xla else to_device(trim_padding(batch), device)
        precision = (torch.autocast(device_type='cuda') if training and device.type == 'cuda'
                     else contextlib.nullcontext())
        with torch.set_grad_enabled(training), precision:
            output = model(*model_inputs(batch), return_details=True)
            loss = batch_loss(batch, output, bone_weight, relative_weight, error_weight, presence_weight, stage_weight)
            if reprojection_weight:
                term, angle, count = reprojection_loss(model, batch, reprojection_queries)
                loss = loss+reprojection_weight*term
                reprojection.add(angle/count.clamp_min(1), count)
        if xla:
            nonfinite = nonfinite+(~torch.isfinite(loss.detach())).int()
        elif not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        if training:
            finite = optimizer_step(model, optimizer, scaler, loss, xla)
            if xla:
                nonfinite = nonfinite+(~finite).int()
            elif finite:
                consecutive = 0
            else:
                skipped, consecutive = skipped+1, consecutive+1
                if consecutive >= MAX_SKIPPED_STEPS:
                    raise FloatingPointError(f'Non-finite gradients in {consecutive} consecutive steps '
                                             f'(grad scale {scaler.get_scale():.3g}): not an fp16 overflow')
        pose.add(output.pose, batch)
        if output.stages is not None:
            coarse.add(output.stages[0], batch)
        loss_mean.add(loss.detach().float(), len(output.pose))
        final = {key: batch[key][:, -1:] if key != 'world_unit_cm' else batch[key]
                 for key in ('target', 'target_mask', 'world_unit_cm')}
        mm = lambda xyz: torch.expm1(joint_error_log(xyz[:, -1:].detach(), final)[0][:, 0])  # [B,2,21]
        actual = mm(output.pose)
        has_target = batch['target_mask'][:, -1].bool()
        if output.error is not None:
            miss = (torch.expm1(output.error[:, -1].detach().float())-actual).abs()
            error_miss.add((miss*has_target).sum()/has_target.sum().clamp_min(1), has_target.sum())
        in_view = batch['hand_in_view'][:, -1].bool()
        out_of_view.add((~in_view).float().mean(), in_view.numel())
        if output.presence is not None:
            presence_hit.add(((output.presence[:, -1] > 0) == in_view).float().mean(), in_view.numel())
        if output.anchored is not None:                     # HandLiteV3; HandDirect has no anchor
            alone = mm(output.anchor_xyz)
            kind = output.anchored[:, -1]
            for index, name in enumerate(ANCHOR_KINDS):
                chosen = ((kind == index) & has_target).float()
                kind_rate[name].add(chosen.sum()/has_target.sum().clamp_min(1), has_target.sum())
                kind_mm[name].add((actual*chosen).sum()/chosen.sum().clamp_min(1), chosen.sum())
                if index:
                    kind_anchor_mm[name].add((alone*chosen).sum()/chosen.sum().clamp_min(1), chosen.sum())
        sync(device)
        batches += 1
        if xla and batches == 1:
            _check_finite(nonfinite, phase)
            print(f'{phase}: first XLA step (compile + run) {time.monotonic()-started:.1f}s', flush=True)
        if batches % log_every == 0:
            if xla:
                _check_finite(nonfinite, phase)
            print(f"{phase} batch={batches} samples={pose.samples} "
                  f"MPJPE={pose.mpjpe_mm:.2f}mm" + (f' skipped_steps={skipped}' if skipped else ''), flush=True)
        if limit and batches >= limit:
            break
    if xla:
        _check_finite(nonfinite, phase)
    summary = pose.summary()
    return dict(loss=loss_mean.mean, mpjpe_mm=summary['mpjpe_mm'], pck20=summary['pck20'],
                mpjpe_world_mm=summary['mpjpe_world_mm'], mpjpe_rel_mm=summary['mpjpe_rel_mm'],
                coarse_mpjpe_mm=coarse.summary()['mpjpe_mm'] if coarse.samples else None,
                reprojection_mrad=None if reprojection.mean is None else reprojection.mean*1e3,
                error_miss_mm=error_miss.mean, presence_accuracy=presence_hit.mean, out_of_view_rate=out_of_view.mean,
                kind_rate={name: mean.mean for name, mean in kind_rate.items()},
                kind_mpjpe_mm={name: mean.mean for name, mean in kind_mm.items()},
                kind_anchor_mpjpe_mm={name: mean.mean for name, mean in kind_anchor_mm.items()},
                samples=summary['samples'], joints=summary['joints'],
                skipped_steps=skipped, batches=batches, seconds=time.monotonic()-started)
