"""Input layout, units and tolerances shared by data, model and stream."""

NUM_CAMERAS = 3
NUM_HANDS = 2
NUM_JOINTS = 21
# One camera frame gives one token per joint of both hands.
HAND_JOINTS = NUM_HANDS*NUM_JOINTS
# Joint order: wrist 0, then four joints per finger (thumb 1-4, ..., little 17-20).
# A finger group is the wrist and that finger's joints.
FINGERS = 5
FINGER_JOINTS = tuple((0,)+tuple(range(1+4*f, 5+4*f)) for f in range(FINGERS))

# Exported per-joint feature channels [...,FEATURE_CHANNELS]. Channels 8-9 (capture and
# arrival relative to the export query) and 11-13 (ids) are not model inputs: the model
# uses the event's own capture/arrival times instead.
FEATURE_CHANNELS = 14
MODEL_CHANNELS = 11
UV = slice(0, 2)               # normalised pixel coordinates in [0,1]
RAY_ORIGIN = slice(2, 5)       # nominal camera centre, world units
RAY_DIRECTION = slice(5, 8)    # nominal ray direction, world axes
DELAY = slice(10, 11)          # capture->arrival delay, seconds

# Learned features express times, ages and delays in this unit so they are O(1).
TIME_UNIT_S = .1
# Ray-to-point miss vectors and ray shifts (world units / unit directions) are scaled by this,
# and angular residuals (radians) divided by RESIDUAL_UNIT (about 3 px of detector noise), to O(1).
MISS_SCALE = 10.
RESIDUAL_UNIT = .01
# Per-camera calibration (the data's auxiliary target, unused by this model): rotation vector (3) and centre shift (3).
CALIBRATION_PARAMS = 6

# An event is available to a query when it arrived by the query time, up to float32
# rounding of times relative to the batch origin.
ARRIVAL_TOLERANCE_S = 1e-6
# The triangulation input's line fit needs samples spread over at least this time std for a
# slope (otherwise their mean), and two rays at least this far from parallel: sin^2 of the
# widest angle between them; 2e-3 is about 2.6 degrees.
MIN_TIME_STD_S = .005
MIN_RAY_SIN2 = 2e-3
# Additive attention bias of an excluded key; finite so fp16 runtimes stay NaN-free.
MASK_OFF = -1e4
# Streams keep events this long beyond the model's context before pruning.
STREAM_KEEP_MARGIN_S = .5


def check_event(camera, features_shape, valid_shape):
    """Validate one camera frame for a stream; returns the camera index."""
    if int(camera) != camera or not 0 <= int(camera) < NUM_CAMERAS:
        raise ValueError(f'camera must be an integer in [0, {NUM_CAMERAS}), got {camera!r}')
    if tuple(features_shape) != (NUM_HANDS, NUM_JOINTS, FEATURE_CHANNELS) or tuple(valid_shape) != (NUM_HANDS, NUM_JOINTS):
        raise ValueError(f'Expected features [2,21,14] and valid [2,21], got {tuple(features_shape)} and {tuple(valid_shape)}')
    return int(camera)
