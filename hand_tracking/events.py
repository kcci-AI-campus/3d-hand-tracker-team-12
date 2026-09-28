"""Camera events: acceptance, latest-ray selection, triangulation samples and the
query-time anchor of HandLite. No learned parameters."""
import torch
import torch.nn.functional as F
from .constants import (NUM_CAMERAS, NUM_HANDS, NUM_JOINTS, HAND_JOINTS, MODEL_CHANNELS, RAY_ORIGIN,
                        RAY_DIRECTION, MISS_SCALE, TIME_UNIT_S, ARRIVAL_TOLERANCE_S, REPEAT_TOLERANCE_S,
                        MIN_TIME_STD_S, MAX_EXTRAPOLATION_S, MIN_RAY_DEPTH)
from .contracts import EventBatch, RawEvents
from .geometry import triangulate, robust_triangulate, apply_calibration



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


def select_slots(events, query, per_camera, span_s):
    """Slots [B,Q,3K]: event indices in arrival order (padding first), and validity. Each
    camera's K latest accepted events that arrived by the query and were captured within
    span_s before it: a fixed-size input whatever the frame rate."""
    b, e = events['camera'].shape
    index = torch.arange(e, device=query.device)
    usable = (events['present'][:, None] & (events['arrival'][:, None] <= query[..., None]+ARRIVAL_TOLERANCE_S)
              & (events['capture'][:, None] >= query[..., None]-span_s))                    # [B,Q,E]
    chosen = []
    for camera in range(NUM_CAMERAS):
        score = torch.where(usable & (events['camera'][:, None] == camera), index, -1)
        if e < per_camera:
            score = F.pad(score, (0, per_camera-e), value=-1)
        chosen.append(score.topk(per_camera, dim=-1).values)
    slots = torch.cat(chosen, -1).sort(-1).values
    return slots.clamp_min(0), slots >= 0


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


def sample_points(origins, directions, mask, capture, params=None, outlier_ratio=None):
    """Triangulate each target's latest-ray sample, optionally calibrated with that
    target's params [B,T,3,6]. Returns point [B,T,2,21,3], ok [B,T,2,21], stamp (mean
    capture of the used rays) [B,T,2,21] and the calibrated rays. outlier_ratio drops one
    inconsistent ray (geometry.robust_triangulate)."""
    b, t = origins.shape[:2]
    if params is not None:
        origins, directions = apply_calibration(params.reshape(b*t,3,6), origins.reshape(b*t,1,*origins.shape[2:]),
                                                directions.reshape(b*t,1,*directions.shape[2:]))
        origins, directions = origins.reshape(b,t,*origins.shape[2:]), directions.reshape(b,t,*directions.shape[2:])
    origins = torch.where(mask[...,None], origins, 0.); directions = torch.where(mask[...,None], directions, 0.)
    order = (0,1,3,4,2,5)
    used = mask.permute(0,1,3,4,2)
    if outlier_ratio:
        point, ok, used = robust_triangulate(origins.permute(order), directions.permute(order), used, outlier_ratio)
    else:
        point, ok = triangulate(origins.permute(order), directions.permute(order), used)
    weight = used.float()
    stamp = (capture[:,:,None,None,:]*weight).sum(-1)/weight.sum(-1).clamp_min(1)
    return point, ok, stamp, (origins, directions)


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


def query_anchor(point, ok, stamp, arrival, present, query, lookback_s, hold_s, single=None):
    """Anchor at each query time from the samples of events that arrived by then.

    point/ok/stamp [B,E,2,21,...] per event sample, arrival/present [B,E], query [B,Q].
    A line fitted to the distinct samples captured within lookback_s is extrapolated to
    the query (at most MAX_EXTRAPOLATION_S); otherwise the latest sample within hold_s is
    held.
    single = (origin, direction, has, capture, reference, known) (see single_ray_anchor): a
    joint the latest event did not triangulate but that has a ray is anchored on the ray
    closest to its anchor, else its reference, else its hand's mean reference, at that depth.
    Joints without a sample use the mean anchor of the same hand, else the origin.
    Returns anchor [B,Q,2,21,3] and flags [B,Q,2,21,5]: latest event's sample valid, joint
    anchor exists, hand anchor exists, age of the latest sample (TIME_UNIT_S), extrapolated;
    with single, a sixth flag marks single-ray anchors (age is then the ray's).
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
    extra = []
    if single is not None:
        anchor, joint, moving, age, ray = single_ray_anchor(anchor, joint, moving, age, now, query, *single)
        extra = [ray.float()]
    weight = joint[...,None].float()
    hand_mean = (anchor*weight).sum(-2, keepdim=True)/weight.sum(-2, keepdim=True).clamp_min(1)
    hand = joint.any(-1, keepdim=True).expand_as(joint)
    anchor = torch.where(joint[...,None], anchor, hand_mean)
    return anchor, torch.stack((now.float(), joint.float(), hand.float(), age/TIME_UNIT_S, moving.float(), *extra), -1)


def single_ray_anchor(anchor, joint, moving, age, now, query, origin, direction, has, capture, reference, known):
    """Anchor joints the latest event did not triangulate (now False) on one of their rays.
    origin/direction [B,Q,2,21,C,3], has/capture [B,Q,2,21,C]: each camera's latest ray;
    reference [B,Q,2,21,3]/known [B,Q,2,21]: a remembered triangulation. The reference point
    is the joint's anchor if it has one, else its remembered triangulation, else the mean of
    its hand's; the ray with the smallest angular residual to it is used (with two rays that
    do not agree, the one not contradicting the past), at the reference's depth. Without a
    reference the joint keeps its anchor."""
    reference = torch.where(joint[...,None], anchor, reference)
    known = joint | known
    weight = known[...,None].float()
    hand = (reference*weight).sum(-2, keepdim=True)/weight.sum(-2, keepdim=True).clamp_min(1)
    reference = torch.where(known[...,None], reference, hand)
    known = known.any(-1, keepdim=True).expand_as(known)
    direction = F.normalize(direction, dim=-1)
    relative = reference[...,None,:]-origin
    along = (relative*direction).sum(-1)                                                # [B,Q,2,21,C]
    residual = (relative-along[...,None]*direction).norm(dim=-1)/along.clamp_min(1e-6)
    best = torch.where(has, residual, float('inf')).argmin(-1, keepdim=True)
    pick = lambda value: value.gather(-1, best).squeeze(-1)
    origin, direction = (value.gather(-2, best[...,None].expand(*best.shape, 3)).squeeze(-2)
                         for value in (origin, direction))
    ray = has.any(-1) & ~now & known
    depth = pick(along)[...,None].clamp_min(MIN_RAY_DEPTH)
    anchor = torch.where(ray[...,None], origin+direction*depth, anchor)
    age = torch.where(ray, query[...,None,None]-pick(capture), age)
    return anchor, joint | ray, moving & ~ray, age, ray


