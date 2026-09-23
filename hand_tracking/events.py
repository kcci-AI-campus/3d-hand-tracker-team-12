"""Camera events: acceptance, latest-ray selection, triangulation samples and the
query-time anchor. No learned parameters; shared by the model and the baseline."""
import math
import torch
import torch.nn.functional as F
from .config import ModelConfig
from .constants import (NUM_CAMERAS, NUM_HANDS, NUM_JOINTS, HAND_JOINTS, MODEL_CHANNELS, RAY_ORIGIN,
                        RAY_DIRECTION, MISS_SCALE, TIME_UNIT_S, ARRIVAL_TOLERANCE_S, REPEAT_TOLERANCE_S,
                        MIN_TIME_STD_S, MAX_EXTRAPOLATION_S)
from .contracts import EventBatch, RawEvents
from .geometry import triangulate, apply_calibration, calibrate_cameras, motion_triangulate

# Angular ray noise the Gauss-Newton baseline calibration assumes (radians).
BASELINE_RAY_NOISE = 1.5e-3


def accepted_captures(camera, capture, present):
    """Keep strictly newer captures per camera in arrival order; padding is ignored.

    This is the batch equivalent of EventStream's per-camera high-water mark: an event is
    dropped when an earlier present event of its camera has a capture at least as new.
    Pairwise [B,E,E] instead of a running maximum (cummax has no ONNX operator).
    Comparing the original dtype preserves float64 stream timestamp precision.
    """
    present = present.bool() & (camera >= 0) & (camera < NUM_CAMERAS) & torch.isfinite(capture)
    index = torch.arange(camera.shape[1], device=camera.device)
    earlier = index[None,:,None] > index[None,None,:]                                   # [1,E(i),E(j)]: j before i
    superseded = (earlier & present[:,None,:] & (camera[:,:,None] == camera[:,None,:])
                  & (capture[:,None,:] >= capture[:,:,None])).any(-1)
    return present & ~superseded


def make_events(features, valid, camera, capture, arrival, present) -> EventBatch:
    present = accepted_captures(camera, capture, present)
    return _pack_events(features, valid, camera, capture, arrival, present)


def _pack_events(features, valid, camera, capture, arrival, present) -> EventBatch:
    """Canonical tensor layout after the caller has resolved event acceptance."""
    valid = valid.bool() & present[...,None,None]
    raw = torch.where(valid[...,None], features[...,:MODEL_CHANNELS].float(), 0.)
    return dict(raw=raw, valid=valid, camera=camera.long(), capture=capture.float(), arrival=arrival.float(),
                present=present.bool())


def relative_events(raw: RawEvents, origin: float) -> EventBatch:
    """Convert one stream's absolute float64 timestamps into a relative model batch."""
    present = torch.ones_like(raw['camera'], dtype=torch.bool)[None]
    # Stream buffers contain only accepted frames; do not rescan history per query.
    return _pack_events(raw['features'][None], raw['valid'][None], raw['camera'][None],
                       (raw['capture']-origin)[None], (raw['arrival']-origin)[None], present)


def _gather(values, index):
    """values [B,E,...], index [B,...] -> values at index along the event axis."""
    batch = torch.arange(values.shape[0], device=values.device).reshape(-1, *[1]*(index.ndim-1))
    return values[batch, index]


def _up_to(events, targets):
    """[B,T,E]: events at or before each target in input (arrival) order. Input order,
    not arrival-time comparison, defines causality, so events with equal arrival times
    see each other exactly as when pushed one by one."""
    index = torch.arange(events['camera'].shape[1], device=events['camera'].device)
    return events['present'][:,None] & (index[None,None] <= targets[None,:,None])


def latest_rays(events, targets, max_age):
    """Each camera's most recently captured ray (among events up to the target, captured
    within max_age of its arrival) for every target event: origins/directions
    [B,T,3,2,21,3], mask [B,T,3,2,21], capture [B,T,3]. A late, older capture never
    replaces a newer one. Frames without any detected joint still count as the latest."""
    arrival = events['arrival']; capture = events['capture']
    target_arrival = arrival[:,targets]
    ok = _up_to(events, targets) & (capture[:,None] >= target_arrival[...,None]-max_age)     # [B,T,E]
    scores = torch.stack([torch.where(ok & (events['camera'][:,None] == c), capture[:,None], float('-inf'))
                          for c in range(NUM_CAMERAS)], -1)                              # [B,T,E,3]
    has = torch.isfinite(scores.amax(2)); pick = scores.argmax(2)                        # [B,T,3]
    raw = _gather(events['raw'], pick)                                                   # [B,T,3,2,21,C]
    mask = _gather(events['valid'], pick) & has[...,None,None]
    return raw[...,RAY_ORIGIN], raw[...,RAY_DIRECTION], mask, torch.where(has, _gather(capture, pick), 0.)


def sample_points(origins, directions, mask, capture, params=None, gate=None):
    """Triangulate each target's latest-ray sample, optionally calibrated with that
    target's params [B,T,3,6]. Returns point [B,T,2,21,3], ok [B,T,2,21], stamp (mean
    capture of the used rays) [B,T,2,21] and the calibrated rays."""
    b, t = origins.shape[:2]
    if params is not None:
        origins, directions = apply_calibration(params.reshape(b*t,3,6), origins.reshape(b*t,1,*origins.shape[2:]),
                                                directions.reshape(b*t,1,*directions.shape[2:]))
        origins, directions = origins.reshape(b,t,*origins.shape[2:]), directions.reshape(b,t,*directions.shape[2:])
    origins = torch.where(mask[...,None], origins, 0.); directions = torch.where(mask[...,None], directions, 0.)
    order = (0,1,3,4,2,5)
    point, ok = triangulate(origins.permute(order), directions.permute(order), mask.permute(0,1,3,4,2), gate)
    weight = mask.permute(0,1,3,4,2).float()
    stamp = (capture[:,:,None,None,:]*weight).sum(-1)/weight.sum(-1).clamp_min(1)
    return point, ok, stamp, (origins, directions)


def _own(values, camera):
    """values [B,T,3,...] per camera slot -> the slot of each target's own camera [B,T,...]."""
    index = camera.reshape(*camera.shape, 1, *[1]*(values.ndim-3)).expand(*camera.shape, 1, *values.shape[3:])
    return values.gather(2, index).squeeze(2)


def _miss(point, ok, origin, direction, use):
    """Perpendicular vectors (x MISS_SCALE) from each ray to the point, zero where unusable."""
    relative = point-origin
    miss = relative-direction*(relative*direction).sum(-1, keepdim=True)
    return torch.where((use & ok)[...,None], miss*MISS_SCALE, 0.)


def _gather_joint(values, pick):
    """values [B,E,2,21,C], pick [B,Q,2,21] event index per joint -> [B,Q,2,21,C]."""
    b, e = values.shape[:2]; q = pick.shape[1]; c = values.shape[-1]
    flat = values.reshape(b, e, HAND_JOINTS, c).permute(0,2,1,3)                        # [B,42,E,C]
    index = pick.reshape(b, q, HAND_JOINTS).permute(0,2,1)[...,None].expand(-1,-1,-1,c) # [B,42,Q,C]
    return flat.gather(2, index).permute(0,2,1,3).reshape(b, q, NUM_HANDS, NUM_JOINTS, c)


def query_anchor(point, ok, stamp, arrival, present, query, lookback_s, hold_s, motion=None):
    """Anchor at each query time from the samples of events that arrived by then.

    point/ok/stamp [B,E,2,21,...] per event sample, arrival/present [B,E], query [B,Q].
    A line fitted to the distinct samples captured within lookback_s is extrapolated to
    the query (at most MAX_EXTRAPOLATION_S); otherwise the latest sample within hold_s is
    held. motion = (position, valid, age) from motion_triangulate overrides where valid.
    Joints without a sample use the mean anchor of the same hand, else the origin.
    Returns anchor [B,Q,2,21,3] and flags [B,Q,2,21,5]: latest event's sample valid, joint
    anchor exists, hand anchor exists, age of the latest sample (TIME_UNIT_S), extrapolated.
    """
    e = arrival.shape[1]; index = torch.arange(e, device=arrival.device)
    available = present[:,None] & (arrival[:,None] <= query[...,None]+ARRIVAL_TOLERANCE_S)  # [B,Q,E]
    latest_event = torch.where(available, index, -1).amax(-1)                          # [B,Q]
    now = _gather(ok, latest_event.clamp_min(0)) & (latest_event >= 0)[...,None,None]
    tau = stamp[:,None]-query[:,:,None,None,None]                                      # [B,Q,E,2,21]
    usable = available[...,None,None] & ok[:,None] & (-tau <= hold_s)
    latest = torch.where(usable, index[None,None,:,None,None], -1).amax(2)             # [B,Q,2,21]
    joint = latest >= 0; pick = latest.clamp_min(0)
    anchor = _gather_joint(point, pick)
    age = torch.where(joint, query[...,None,None]-_gather_joint(stamp[...,None], pick)[...,0], 0.)
    moving = torch.zeros_like(joint)
    if lookback_s > 0:
        use = usable & (-tau <= lookback_s)
        # A sample repeated over events (a joint the newest event did not see) counts once.
        same = ((stamp[:,:,None]-stamp[:,None]).abs() <= REPEAT_TOLERANCE_S) & ok[:,:,None] & ok[:,None]
        count = torch.einsum('bsuhj,btuhj->btshj', same.float(), use.float())
        w = torch.where(use, 1/count.clamp_min(1), 0.)
        total = w.sum(2)
        mean_tau = (w*tau).sum(2)/total.clamp_min(1e-12)
        mean_point = (w[...,None]*point[:,None]).sum(2)/total.clamp_min(1e-12)[...,None]
        centred = tau-mean_tau[:,:,None]
        var_tau = (w*centred.square()).sum(2)
        slope = (w[...,None]*centred[...,None]*(point[:,None]-mean_point[:,:,None])).sum(2)
        slope = slope/var_tau.clamp_min(1e-12)[...,None]
        moving = (total >= 2-1e-6) & (var_tau > MIN_TIME_STD_S**2*total)
        anchor = torch.where(moving[...,None], mean_point+slope*(-mean_tau).clamp(0., MAX_EXTRAPOLATION_S)[...,None], anchor)
    if motion is not None:
        position, fitted, fitted_age = motion
        anchor = torch.where(fitted[...,None], position, anchor)
        joint = joint | fitted; moving = moving | fitted; age = torch.where(fitted, fitted_age, age)
    weight = joint[...,None].float()
    hand_mean = (anchor*weight).sum(-2, keepdim=True)/weight.sum(-2, keepdim=True).clamp_min(1)
    hand = joint.any(-1, keepdim=True).expand_as(joint)
    anchor = torch.where(joint[...,None], anchor, hand_mean)
    return anchor, torch.stack((now.float(), joint.float(), hand.float(), age/TIME_UNIT_S, moving.float()), -1)


def _event_motion(events, own_origin, own_direction, query, lookback_s):
    """Motion-fit inputs: each event's own calibrated rays in its camera slot."""
    b, e = events['camera'].shape
    slot = F.one_hot(events['camera'], NUM_CAMERAS).bool()[...,None,None] & events['valid'][:,:,None]   # [B,E,3,2,21]
    spread = lambda v: torch.where(slot[...,None], v[:,:,None], 0.)
    capture = torch.where(slot, events['capture'][:,:,None,None,None], 0.)
    span = events['present'][:,None] & (events['arrival'][:,None] <= query[...,None]+ARRIVAL_TOLERANCE_S)
    return motion_triangulate(spread(own_origin), spread(own_direction), slot, capture, query, span, lookback_s)


def event_anchor(features, valid, camera, capture, arrival, present, query,
                 lookback_s=ModelConfig.anchor_lookback_s, hold_s=ModelConfig.anchor_hold_s,
                 sample_max_age_s=ModelConfig.sample_max_age_s, calibration_steps=0,
                 gate=ModelConfig.anchor_ray_gate, motion_fit=ModelConfig.anchor_motion_fit,
                 rotation_deg=ModelConfig.calibration_rotation_deg, position_std=ModelConfig.calibration_position):
    """Model-free anchor baseline [B,Q,2,21,3] with nominal or per-window Gauss-Newton calibration
    (prior std rotation_deg/position_std, as the learned head's scale)."""
    events = make_events(features, valid, camera, capture, arrival, present)
    targets = torch.arange(camera.shape[1], device=camera.device)
    o, d, mask, cap = latest_rays(events, targets, sample_max_age_s)
    params = None
    if calibration_steps:
        with torch.no_grad():
            window = calibrate_cameras(o, d, mask, calibration_steps, math.radians(rotation_deg), position_std,
                                       BASELINE_RAY_NOISE)
        params = window[:,None].expand(-1, camera.shape[1], -1, -1)
    point, ok, stamp, (co, cd) = sample_points(o, d, mask, cap, params, gate)
    ok = ok & events['present'][...,None,None]
    motion = None
    if motion_fit and lookback_s > 0:
        own = events['camera'][:,targets]
        motion = _event_motion(events, _own(co, own), _own(cd, own), query.float(), lookback_s)
    return query_anchor(point, ok, stamp, events['arrival'], events['present'], query.float(), lookback_s, hold_s, motion)[0]
