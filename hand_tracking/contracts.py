"""Tensor contracts shared by loaders, model, and streaming inference.

Model times are float32 seconds relative to a nearby origin. Stream storage keeps
absolute timestamps in float64. Shapes below include the batch dimension.
"""
from dataclasses import dataclass
from typing import Callable, NamedTuple, TypedDict
from torch import Tensor
import torch
from .constants import HAND_JOINTS

MODEL_INPUT_KEYS = ('event_features', 'event_valid', 'event_camera', 'event_capture',
                    'event_arrival', 'event_present', 'query_times')


# Supervision and unit conversion; never model inputs.
TARGET_KEYS = ('target', 'target_mask', 'calibration_target', 'world_unit_cm')


def model_inputs(batch):
    return tuple(batch[key] for key in MODEL_INPUT_KEYS)


class WindowSample(TypedDict):
    """One training/evaluation window (data.make_sample); a DataLoader adds a batch dim.
    Times are float32 seconds relative to the window's last query."""
    event_features: Tensor      # [E,2,21,14], invalid joints zeroed
    event_valid: Tensor         # [E,2,21]
    event_camera: Tensor        # [E] long
    event_present: Tensor       # [E], False = padding
    event_capture: Tensor       # [E]
    event_arrival: Tensor       # [E]
    query_times: Tensor         # [Q]
    target: Tensor              # [Q,2,21,3] world units
    target_mask: Tensor         # [Q,2,21]
    calibration_target: Tensor  # [3,6], NaN when unknown (real captures)
    world_unit_cm: Tensor       # []


WINDOW_SAMPLE_KEYS = tuple(WindowSample.__annotations__)
assert set(WINDOW_SAMPLE_KEYS) == set(MODEL_INPUT_KEYS+TARGET_KEYS)


class EventBatch(TypedDict):
    raw: Tensor       # [B,E,2,21,MODEL_CHANNELS], invalid joints zeroed
    valid: Tensor     # [B,E,2,21]
    camera: Tensor    # [B,E]
    capture: Tensor   # [B,E], relative seconds
    arrival: Tensor   # [B,E], relative seconds
    present: Tensor   # [B,E], excludes padding and stale captures


class RawEvents(TypedDict):
    features: Tensor
    valid: Tensor
    camera: Tensor
    capture: Tensor   # stream: [E], absolute float64 seconds
    arrival: Tensor


class ModelOutput(NamedTuple):
    """HandTransformer(..., return_details=True)."""
    pose: Tensor         # [B,Q,2,21,3] world units
    calibration: Tensor  # [B,E,3,6] per-event camera correction (see geometry.apply_calibration)
    accepted: Tensor     # [B,E] events the model used: present and newer than that camera's prior capture


class LayerKV(NamedTuple):
    key: Tensor       # [B,E,42,heads,head_dim]
    value: Tensor


@dataclass
class Evidence:
    summary: Tensor   # [B,E,dim]
    evident: Tensor   # [B,E]

    def map(self, fn: Callable[[Tensor], Tensor]):
        return Evidence(fn(self.summary), fn(self.evident))

    def append(self, other, dim=1):
        return Evidence(torch.cat((self.summary, other.summary), dim),
                        torch.cat((self.evident, other.evident), dim))


@dataclass
class EncodedEvents:
    point: Tensor     # [B,E,2,21,3]
    ok: Tensor        # [B,E,2,21], triangulation validity
    stamp: Tensor     # [B,E,2,21], mean capture time of triangulated rays
    keys: tuple[LayerKV, ...]
    origin: Tensor
    direction: Tensor

    def map(self, fn: Callable[[Tensor], Tensor]):
        """Apply a batch/index/device transform consistently to every cached tensor."""
        return EncodedEvents(fn(self.point), fn(self.ok), fn(self.stamp),
                             tuple(LayerKV(fn(k), fn(v)) for k, v in self.keys),
                             fn(self.origin), fn(self.direction))

    def append(self, other, dim=1):
        def cat(a, b):
            return torch.cat((a, b), dim)
        return EncodedEvents(cat(self.point, other.point), cat(self.ok, other.ok),
                             cat(self.stamp, other.stamp),
                             tuple(LayerKV(cat(a.key, b.key), cat(a.value, b.value))
                                   for a, b in zip(self.keys, other.keys, strict=True)),
                             cat(self.origin, other.origin), cat(self.direction, other.direction))


def to_joint_sequences(value):
    """[B,Q,42,D] -> [B*42,Q,D] for attention along each joint's event history."""
    batch, queries, joints, width = value.shape
    return value.permute(0, 2, 1, 3).reshape(batch*joints, queries, width)


def to_pose_sequences(value, batch):
    """[B*42,Q,D] -> [B,Q,42,D]."""
    return value.reshape(batch, HAND_JOINTS, value.shape[1], value.shape[2]).permute(0, 2, 1, 3)
