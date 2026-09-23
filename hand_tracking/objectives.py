"""Training losses (pose, bone length, calibration auxiliary) and evaluation totals."""
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


def error_totals(prediction, target, mask, world_unit_cm):
    """MPJPE numerator (mm), PCK@20mm numerator and valid joint count for [B,2,21,3]."""
    valid = mask.bool()
    distance = (prediction-torch.where(valid[...,None],target,0.)).norm(dim=-1)
    mm = distance*world_unit_cm[:,None,None]*10
    return (mm*valid).sum(), ((mm<20)&valid).sum(), valid.sum()


@dataclass
class PoseTotals:
    """Running MPJPE and PCK@20mm over the final query of each window (the reported metric)."""
    error_mm: float = 0.
    correct: int = 0
    joints: int = 0
    samples: int = 0

    def add(self, pose, batch):
        """pose [B,Q,2,21,3] against a batch's target/target_mask/world_unit_cm."""
        with torch.no_grad():
            error, correct, joints = error_totals(pose[:,-1].float(), batch['target'][:,-1],
                                                  batch['target_mask'][:,-1], batch['world_unit_cm'])
        self.error_mm += error.item()
        self.correct += correct.item()
        self.joints += joints.item()
        self.samples += len(pose)

    @property
    def mpjpe_mm(self):
        return self.error_mm/self.joints if self.joints else float('nan')

    def summary(self):
        if not self.joints:
            raise ValueError('No valid observations/targets in this loader')
        return dict(mpjpe_mm=self.mpjpe_mm, pck20=self.correct/self.joints, samples=self.samples, joints=self.joints)


@dataclass
class WeightedMean:
    total: float = 0.
    weight: float = 0.

    def add(self, value, weight):
        self.total += value*weight
        self.weight += weight

    @property
    def mean(self):
        """None when nothing was weighted (e.g. no known calibration target)."""
        return self.total/self.weight if self.weight else None
