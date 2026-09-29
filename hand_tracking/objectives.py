"""Training losses (pose, wrist-relative pose, bone length, calibration auxiliary) and evaluation totals."""
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F


def calibration_loss(prediction, target, rotation_deg=5., position_std=.1):
    """Mean squared calibration error in prior-std units over cameras with a known target.

    prediction/target [N,3,6]; target rows of NaN (e.g. real captures, padding) are ignored.
    The common rig motion is unobservable, so the optimum is its posterior mean.
    """
    scale = prediction.new_tensor([math.radians(rotation_deg)]*3+[position_std]*3)
    known = torch.isfinite(target).all(-1)
    error = ((prediction-torch.where(known[...,None], target, 0.))/scale).square().mean(-1)
    return (error*known).sum()/known.sum().clamp_min(1)


BONES = [(0,1+4*f) for f in range(5)] + [(1+4*f+j,2+4*f+j) for f in range(5) for j in range(3)]


def pose_loss(prediction, target, target_mask, world_unit_cm, bone_weight=.1, beta_m=.006):
    """Masked SmoothL1 plus target-specific bone length loss, in meters.

    prediction/target [B,...,2,21,3]; world_unit_cm [B] converts each sample, so
    beta and the loss scale do not depend on the export's world unit.
    """
    valid = target_mask.bool()
    scale = (world_unit_cm/100).reshape(-1, *[1]*(prediction.ndim-1))
    prediction = prediction.float()*scale
    safe_target = torch.where(valid[...,None], target, 0.)*scale
    joints = F.smooth_l1_loss(prediction, safe_target, beta=beta_m, reduction='none').mean(-1)
    position = (joints*valid).sum()/valid.sum().clamp_min(1)
    a, b = map(list, zip(*BONES))
    pred_lengths = (prediction[...,a,:]-prediction[...,b,:]).norm(dim=-1)
    true_lengths = (safe_target[...,a,:]-safe_target[...,b,:]).norm(dim=-1)
    bone_mask = valid[...,a] & valid[...,b]
    bone = ((pred_lengths-true_lengths).abs()*bone_mask).sum()/bone_mask.sum().clamp_min(1)
    return position+bone_weight*bone


def wrist_relative(joints, mask):
    """Joints [...,21,3] and mask [...,21] -> each non-wrist joint minus its hand's wrist
    [...,20,3] and where both have a target [...,20]."""
    return joints[...,1:,:]-joints[...,:1,:], mask[...,1:] & mask[...,:1]


def relative_loss(prediction, target, target_mask, world_unit_cm, beta_m=.006):
    """Masked SmoothL1 of the joints relative to their wrist (the hand's shape), in meters.

    The wrist position error cancels, so hand shape keeps its own training signal; used
    beside pose_loss, which keeps the absolute position (and hands without a wrist target).
    """
    valid = target_mask.bool()
    scale = (world_unit_cm/100).reshape(-1, *[1]*(prediction.ndim-1))
    prediction, mask = wrist_relative(prediction.float()*scale, valid)
    safe_target, _ = wrist_relative(torch.where(valid[...,None], target, 0.)*scale, valid)
    joints = F.smooth_l1_loss(prediction, safe_target, beta=beta_m, reduction='none').mean(-1)
    return (joints*mask).sum()/mask.sum().clamp_min(1)


def error_totals(prediction, target, mask, world_unit_cm):
    """MPJPE numerator (mm), PCK@20mm numerator and valid joint count for [B,2,21,3]."""
    valid = mask.bool()
    distance = (prediction-torch.where(valid[...,None],target,0.)).norm(dim=-1)
    mm = distance*world_unit_cm[:,None,None]*10
    return (mm*valid).sum(), ((mm<20)&valid).sum(), valid.sum()


def relative_error_totals(prediction, target, mask, world_unit_cm):
    """Wrist-relative MPJPE numerator (mm) and joint count for [B,2,21,3] (hand shape only)."""
    valid = mask.bool()
    prediction, relative_mask = wrist_relative(prediction, valid)
    target, _ = wrist_relative(torch.where(valid[...,None], target, 0.), valid)
    mm = (prediction-target).norm(dim=-1)*world_unit_cm[:,None,None]*10
    return (mm*relative_mask).sum(), relative_mask.sum()


def _number(value):
    """A running total as a Python number; a device tensor is read (waited for) only here."""
    return value.item() if isinstance(value, torch.Tensor) else value


@dataclass
class PoseTotals:
    """Running MPJPE and PCK@20mm over the final query of each window (the reported metric),
    against the target frame's targets, MPJPE against the world-frame targets, and the
    wrist-relative MPJPE (hand shape, wrist position error removed).

    Totals stay device tensors between reads, so adding a batch never waits for the device
    (under XLA, a read ends the traced step); mpjpe_mm and summary() read them."""
    error_mm: float = 0.
    correct: int = 0
    joints: int = 0
    samples: int = 0
    world_error_mm: float = 0.
    relative_error_mm: float = 0.
    relative_joints: int = 0

    def add(self, pose, batch):
        """pose [B,Q,2,21,3] against a batch's target/target_mask/world_unit_cm."""
        with torch.no_grad():
            error, correct, joints = error_totals(pose[:,-1].float(), batch['target'][:,-1],
                                                  batch['target_mask'][:,-1], batch['world_unit_cm'])
            world, _, _ = error_totals(pose[:,-1].float(), batch['target_world'][:,-1], batch['target_mask'][:,-1],
                                       batch['world_unit_cm'])
            relative, relative_joints = relative_error_totals(pose[:,-1].float(), batch['target'][:,-1],
                                                              batch['target_mask'][:,-1], batch['world_unit_cm'])
        self.error_mm = self.error_mm+error
        self.world_error_mm = self.world_error_mm+world
        self.relative_error_mm = self.relative_error_mm+relative
        self.relative_joints = self.relative_joints+relative_joints
        self.correct = self.correct+correct
        self.joints = self.joints+joints
        self.samples += len(pose)

    @property
    def mpjpe_mm(self):
        joints = _number(self.joints)
        return _number(self.error_mm)/joints if joints else float('nan')

    def summary(self):
        joints, relative_joints = int(_number(self.joints)), int(_number(self.relative_joints))
        if not joints:
            raise ValueError('No valid observations/targets in this loader')
        relative = _number(self.relative_error_mm)/relative_joints if relative_joints else float('nan')
        return dict(mpjpe_mm=_number(self.error_mm)/joints, pck20=_number(self.correct)/joints,
                    mpjpe_world_mm=_number(self.world_error_mm)/joints, mpjpe_rel_mm=relative,
                    samples=self.samples, joints=joints)


@dataclass
class WeightedMean:
    """Weighted running mean; value/weight may be device tensors, read only by mean."""
    total: float = 0.
    weight: float = 0.

    def add(self, value, weight):
        self.total = self.total+value*weight
        self.weight = self.weight+weight

    @property
    def mean(self):
        """None when nothing was weighted (e.g. no known calibration target)."""
        weight = _number(self.weight)
        return _number(self.total)/weight if weight else None
