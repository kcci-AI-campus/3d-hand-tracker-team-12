"""Helpers shared by the model and its export: per-event encoder input, camera one-hot, fixed-count
repeats and the per-event token store."""
from dataclasses import dataclass
import torch
from .constants import NUM_CAMERAS, UV, RAY_ORIGIN, RAY_DIRECTION, DELAY, TIME_UNIT_S

# Per joint: u,v (2), ray origin (3) and direction (3), delay (1), detected (1).
DIRECT_JOINT_FEATURES = 10


def direct_features(raw, valid):
    """Event features [...,2,21,11] (invalid joints zeroed), valid -> encoder input [...,2,21,10]."""
    x = torch.cat((raw[..., UV]*2-1, raw[..., RAY_ORIGIN], raw[..., RAY_DIRECTION], raw[..., DELAY]/TIME_UNIT_S,
                   valid.float()[..., None]), -1)
    return torch.where(valid[..., None], x, 0.)


def camera_one_hot(camera):
    """Camera index [...] -> one-hot [...,3] by comparison (F.one_hot checks the index range
    on the host, a device wait under XLA)."""
    return (camera[..., None] == torch.arange(NUM_CAMERAS, device=camera.device)).float()


def repeat_each(value, count):
    """[...,S] -> [...,S*count], each element repeated count times in place (repeat_interleave
    with a fixed count, as expand + reshape: a static shape under XLA)."""
    return value[..., None].expand(*value.shape, count).reshape(*value.shape[:-1], -1)


@dataclass
class DirectEncoded:
    """Per-event tokens [B,E,...,dim]; they never change after arrival."""
    tokens: torch.Tensor

    def map(self, fn):
        return DirectEncoded(fn(self.tokens))

    def append(self, other, dim=1):
        return DirectEncoded(torch.cat((self.tokens, other.tokens), dim))
