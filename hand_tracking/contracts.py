"""Tensor contracts shared by loaders, model, and streaming inference.

Model times are float32 seconds relative to a nearby origin. Stream storage keeps
absolute timestamps in float64. Shapes below include the batch dimension.
"""
from typing import NamedTuple, TypedDict
from torch import Tensor

MODEL_INPUT_KEYS = ('event_features', 'event_valid', 'event_camera', 'event_capture',
                    'event_arrival', 'event_present', 'query_times')


# Supervision and unit conversion; never model inputs.
TARGET_KEYS = ('target', 'target_world', 'target_mask', 'calibration_target', 'world_unit_cm')


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
    target: Tensor              # [Q,2,21,3] world units, in the sampling target_frame (rig or world)
    target_world: Tensor        # [Q,2,21,3] the same in the simulator's true world frame
    target_mask: Tensor         # [Q,2,21]
    calibration_target: Tensor  # [3,6] in the target frame, NaN when unknown (real captures)
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
    """model(..., return_details=True): calibration per query [B,Q,3,6] (None for a model
    without camera correction); accepted marks the queries with at least one slot."""
    pose: Tensor         # [B,Q,2,21,3] world units
    calibration: Tensor  # [B,Q,3,6] camera correction (see geometry.apply_calibration)
    accepted: Tensor     # [B,Q]
