"""Input layout, units and tolerances shared by data, geometry, model and stream."""

NUM_CAMERAS = 3
NUM_HANDS = 2
NUM_JOINTS = 21
# One camera frame gives one token per joint of both hands.
HAND_JOINTS = NUM_HANDS*NUM_JOINTS

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
# Ray-to-point miss vectors are scaled to O(1) for the same reason.
MISS_SCALE = 10.
# Per-camera calibration: rotation vector (3) and centre shift (3).
CALIBRATION_PARAMS = 6

# An event is available to a query when it arrived by the query time, up to float32
# rounding of times relative to the batch origin.
ARRIVAL_TOLERANCE_S = 1e-6
# Distinct sample stamps are >=1/camera_fps/3 apart; float32 relative times differ by ~1e-7.
REPEAT_TOLERANCE_S = 1e-5
# Velocity needs samples spread over at least this time std; extrapolation is capped.
MIN_TIME_STD_S = .005
MAX_EXTRAPOLATION_S = .25
# Triangulation needs two rays at least this far from parallel: sin^2 of the widest
# angle between them; 2e-3 is about 2.6 degrees.
MIN_RAY_SIN2 = 2e-3
