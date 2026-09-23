"""Compatibility imports for older scripts. New code imports from hand_tracking.*;
checkpoint keys are unchanged."""
from hand_tracking.config import ARRIVAL_MARGIN_S, ModelConfig
from hand_tracking.constants import REPEAT_TOLERANCE_S, MIN_TIME_STD_S, MAX_EXTRAPOLATION_S
from hand_tracking.geometry import (
    MIN_RAY_SIN2, HUBER_RAY_RESIDUAL, MAX_MOTION_RESIDUAL, MAX_MOTION_RMS,
    triangulate, rodrigues, apply_calibration, calibrate_cameras, motion_triangulate,
)
from hand_tracking.events import make_events, latest_rays, sample_points, query_anchor, event_anchor
from hand_tracking.model import CrossAttention, DecoderLayer, HandTransformer
from hand_tracking.stream import EventStream
from hand_tracking.objectives import calibration_loss, BONES, pose_loss, error_totals
