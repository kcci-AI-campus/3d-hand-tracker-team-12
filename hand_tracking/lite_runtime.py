"""HandLite deployment runtime without PyTorch: numpy geometry around the three exported
networks, run by ncnn or ONNX Runtime. Mirrors hand_tracking.lite.LiteStream.

    runtime = LiteRuntime('exports/lite_model', backend='ncnn')
    runtime.push(camera, features [2,21,14], valid [2,21], capture_time, arrival_time)
    pose = runtime.query(time)          # [2,21,3] world units

Times are absolute seconds (float64). Needs numpy plus `ncnn` or `onnxruntime`.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from .config import saved_config
from .constants import (NUM_CAMERAS, NUM_HANDS, NUM_JOINTS, HAND_JOINTS, MODEL_CHANNELS, UV, RAY_ORIGIN,
                        RAY_DIRECTION, DELAY, TIME_UNIT_S, MISS_SCALE, ARRIVAL_TOLERANCE_S, REPEAT_TOLERANCE_S,
                        MIN_TIME_STD_S, MAX_EXTRAPOLATION_S, MIN_RAY_SIN2, MIN_RAY_DEPTH, RAY_RESIDUAL_FLOOR, OUTLIER_MARGIN,
                        MASK_OFF,
                        STREAM_KEEP_MARGIN_S,
                        check_event)

META_FILE = 'lite.json'
# Graph interface version; 2: the decoder takes token validity (0/1), not an additive bias;
# 3: the encoder makes config.event_tokens tokens per event (finger tokens). Version 2
# exports load with the options they predate off (config.LEGACY_OPTIONS); 4: the decoder
# takes config.anchor_features per joint (single-ray anchor flag).
EXPORT_FORMAT = 4
READABLE_FORMATS = (2, 3, 4)
GRAPHS = ('encoder', 'calibrator', 'decoder')


# --------------------------------------------------------------------------- geometry

def rodrigues(w):
    """Rotation vectors [...,3] -> matrices [...,3,3]."""
    theta = np.linalg.norm(w, axis=-1)[..., None, None]
    small = theta < 1e-4
    safe = np.where(small, 1., theta)
    a = np.where(small, 1-theta**2/6, np.sin(safe)/safe)
    half = safe/2
    b = np.where(small, .5-theta**2/24, .5*(np.sin(half)/half)**2)
    x, y, z = w[..., 0], w[..., 1], w[..., 2]
    zero = np.zeros_like(x)
    skew = np.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*w.shape[:-1], 3, 3)
    return np.eye(3)+a*skew+b*(skew@skew)


def triangulate(origins, directions, mask):
    """origins/directions [...,C,3], mask [...,C] -> point [...,3], valid [...] (see geometry.triangulate)."""
    mask = (mask & np.isfinite(origins).all(-1) & np.isfinite(directions).all(-1)
            & (np.linalg.norm(directions, axis=-1) > 1e-8))
    d = np.where(mask[..., None], directions, 0.)
    o = np.where(mask[..., None], origins, 0.)
    d = d/np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-8)
    # sum over rays of (I - d d^T) and (I - d d^T) o; masked rays have d = o = 0.
    count = mask.sum(-1)
    a = count[..., None, None]*np.eye(3)-np.einsum('...ci,...cj->...ij', d, d)
    rhs = (o-d*(d*o).sum(-1, keepdims=True)).sum(-2)
    cameras = d.shape[-2]
    sin2 = np.zeros(count.shape)
    for i in range(cameras):
        for j in range(i+1, cameras):
            sin2 = np.maximum(sin2, np.square(np.cross(d[..., i, :], d[..., j, :])).sum(-1))
    valid = (count >= 2) & (sin2 > MIN_RAY_SIN2)
    a = np.where(valid[..., None, None], a, np.eye(3))
    r0, r1, r2 = a[..., 0, :], a[..., 1, :], a[..., 2, :]
    c0, c1, c2 = np.cross(r1, r2), np.cross(r2, r0), np.cross(r0, r1)
    point = (rhs[..., 0:1]*c0+rhs[..., 1:2]*c1+rhs[..., 2:3]*c2)/(r0*c0).sum(-1, keepdims=True)
    depth = ((point[..., None, :]-o)*d).sum(-1)
    valid = valid & ((depth > 0) | ~mask).all(-1)
    return np.where(valid[..., None], point, 0.), valid


def ray_residuals(point, origins, directions):
    """Angular residual of each ray [...,C] to point [...,3] (see geometry.ray_residuals)."""
    d = directions/np.maximum(np.linalg.norm(directions, axis=-1, keepdims=True), 1e-8)
    relative = point[..., None, :]-origins
    depth = (relative*d).sum(-1)
    return np.linalg.norm(relative-depth[..., None]*d, axis=-1)/np.maximum(depth, 1e-6)


def robust_triangulate(origins, directions, mask, outlier_ratio):
    """triangulate dropping at most one outlier ray (see geometry.robust_triangulate);
    returns point, valid and the rays used."""
    point, valid = triangulate(origins, directions, mask)
    cameras = np.arange(mask.shape[-1])
    several = mask.sum(-1) >= 3
    best, second = np.zeros(valid.shape), np.zeros(valid.shape)
    best_point, best_valid, dropped = point, valid, np.zeros_like(mask)
    for k in cameras:
        keep = mask & (cameras != k)
        loo_point, loo_valid = triangulate(origins, directions, keep)
        residual = ray_residuals(loo_point, origins, directions)
        spread = np.where(keep, residual, 0.).max(-1)
        ratio = np.where(loo_valid & mask[..., k] & several, residual[..., k]/np.maximum(spread, RAY_RESIDUAL_FLOOR), 0.)
        better = ratio > best
        second = np.where(better, best, np.maximum(second, ratio))
        best = np.where(better, ratio, best)
        best_point = np.where(better[..., None], loo_point, best_point)
        best_valid = np.where(better, loo_valid, best_valid)
        dropped = np.where(better[..., None], cameras == k, dropped)
    suspect = best > outlier_ratio
    ambiguous = suspect & (best < OUTLIER_MARGIN*second)
    reject = suspect & ~ambiguous
    valid = np.where(reject, best_valid, valid) & ~ambiguous
    point = np.where(valid[..., None], np.where(reject[..., None], best_point, point), 0.)
    return point, valid, mask & ~(reject[..., None] & dropped)


def calibrate(origins, directions, params):
    """Rays [...,3,2,21,3] corrected by params [3,6] (see geometry.apply_calibration)."""
    rotation = rodrigues(params[:, :3])                                            # [3,3,3]
    return origins+params[:, None, None, 3:], np.einsum('cxy,...chjy->...chjx', rotation, directions)


def sample(origins, directions, mask, capture, params=None, outlier_ratio=0.):
    """Rays [...,3,2,21,3] of the three cameras (optionally corrected by params [3,6]) ->
    point [...,2,21,3], ok [...,2,21], stamp (mean capture of the used rays) [...,2,21];
    outlier_ratio > 0 drops one inconsistent ray (robust_triangulate)."""
    if params is not None:
        origins, directions = calibrate(origins, directions, params)
    # [...,camera,hand,joint,xyz] -> [...,hand,joint,camera,xyz]
    rays = list(range(origins.ndim-4))+[origins.ndim-3, origins.ndim-2, origins.ndim-4, origins.ndim-1]
    masks = list(range(mask.ndim-3))+[mask.ndim-2, mask.ndim-1, mask.ndim-3]
    used = np.transpose(mask, masks)
    if outlier_ratio:
        point, ok, used = robust_triangulate(np.transpose(origins, rays), np.transpose(directions, rays), used,
                                             outlier_ratio)
    else:
        point, ok = triangulate(np.transpose(origins, rays), np.transpose(directions, rays), used)
    weight = used.astype(np.float64)
    stamp = (capture[..., None, None, :]*weight).sum(-1)/np.maximum(weight.sum(-1), 1)
    return point, ok, stamp


def anchor_at(point, ok, stamp, query, lookback_s, hold_s, single=None):
    """Anchor [2,21,3] and flags [2,21,5] at query from slot samples [S,...] in arrival
    order (all arrived by the query); see events.query_anchor. single = (origin, direction,
    has, capture, reference, known) [2,21,...] adds single-ray anchors and a sixth flag."""
    s = len(point)
    if not s:
        # No slot in the span (e.g. a camera outage): origin and all flags off, as in torch.
        return np.zeros((NUM_HANDS, NUM_JOINTS, 3)), np.zeros((NUM_HANDS, NUM_JOINTS, 5+(single is not None)))
    index = np.arange(s)
    now = ok[-1]
    tau = stamp-query
    usable = ok & (-tau <= hold_s)
    latest = np.where(usable, index[:, None, None], -1).max(0)
    joint = latest >= 0
    pick = np.maximum(latest, 0)
    hands, joints = np.indices(joint.shape)
    anchor = point[pick, hands, joints]
    age = np.where(joint, query-stamp[pick, hands, joints], 0.)
    moving = np.zeros_like(joint)
    if lookback_s > 0:
        use = usable & (-tau <= lookback_s)
        same = (np.abs(stamp[:, None]-stamp[None]) <= REPEAT_TOLERANCE_S) & ok[:, None] & ok[None]
        count = (same & use[None]).sum(1)
        w = np.where(use, 1/np.maximum(count, 1), 0.)
        total = w.sum(0)
        mean_tau = (w*tau).sum(0)/np.maximum(total, 1e-12)
        mean_point = (w[..., None]*point).sum(0)/np.maximum(total, 1e-12)[..., None]
        centred = tau-mean_tau
        var_tau = (w*centred**2).sum(0)
        slope = (w[..., None]*centred[..., None]*(point-mean_point)).sum(0)/np.maximum(var_tau, 1e-12)[..., None]
        moving = (total >= 2-1e-6) & (var_tau > MIN_TIME_STD_S**2*total)
        extrapolated = mean_point+slope*np.clip(-mean_tau, 0., MAX_EXTRAPOLATION_S)[..., None]
        anchor = np.where(moving[..., None], extrapolated, anchor)
    extra = []
    if single is not None:
        anchor, joint, moving, age, ray = single_ray_anchor(anchor, joint, moving, age, now, query, *single)
        extra = [ray]
    weight = joint[..., None].astype(np.float64)
    hand_mean = (anchor*weight).sum(-2, keepdims=True)/np.maximum(weight.sum(-2, keepdims=True), 1)
    hand = np.broadcast_to(joint.any(-1, keepdims=True), joint.shape)
    anchor = np.where(joint[..., None], anchor, hand_mean)
    flags = np.stack((now, joint, hand, age/TIME_UNIT_S, moving, *extra), -1).astype(np.float64)
    return anchor, flags


def single_ray_anchor(anchor, joint, moving, age, now, query, origin, direction, has, capture, reference, known):
    """See events.single_ray_anchor."""
    reference = np.where(joint[..., None], anchor, reference)
    known = joint | known
    weight = known[..., None].astype(np.float64)
    hand = (reference*weight).sum(-2, keepdims=True)/np.maximum(weight.sum(-2, keepdims=True), 1)
    reference = np.where(known[..., None], reference, hand)
    known = np.broadcast_to(known.any(-1, keepdims=True), known.shape)
    direction = direction/np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-12)
    relative = reference[..., None, :]-origin
    along = (relative*direction).sum(-1)                                                # [2,21,C]
    residual = np.linalg.norm(relative-along[..., None]*direction, axis=-1)/np.maximum(along, 1e-6)
    best = np.where(has, residual, np.inf).argmin(-1)[..., None]
    pick = lambda value: np.take_along_axis(value, best, -1)[..., 0]
    origin, direction = (np.take_along_axis(value, best[..., None], -2)[..., 0, :] for value in (origin, direction))
    ray = has.any(-1) & ~now & known
    depth = np.maximum(pick(along), MIN_RAY_DEPTH)[..., None]
    anchor = np.where(ray[..., None], origin+direction*depth, anchor)
    age = np.where(ray, query-pick(capture), age)
    return anchor, joint | ray, moving & ~ray, age, ray


def joint_features(raw, valid, point, ok):
    """Event features [2,21,11], valid, nominal sample -> encoder input [2,21,14]."""
    origin, direction = raw[..., RAY_ORIGIN], raw[..., RAY_DIRECTION]
    relative = point-origin
    miss = relative-direction*(relative*direction).sum(-1, keepdims=True)
    miss = np.where((valid & ok)[..., None], miss*MISS_SCALE, 0.)
    x = np.concatenate((raw[..., UV]*2-1, origin, direction, raw[..., DELAY]/TIME_UNIT_S, miss,
                        valid[..., None], (ok & valid)[..., None]), -1)
    return np.where(valid[..., None], x, 0.)


def select_slot_events(events, time, per_camera, span_s):
    """Each camera's per_camera latest events (with .camera/.capture/.arrival, in arrival
    order) arrived by time and captured within span_s, in arrival order; None for padding
    (first). The runtimes' events.select_slots."""
    chosen = [[] for _ in range(NUM_CAMERAS)]
    for index in range(len(events)-1, -1, -1):
        event = events[index]
        if event.arrival <= time+ARRIVAL_TOLERANCE_S and event.capture >= time-span_s \
                and len(chosen[event.camera]) < per_camera:
            chosen[event.camera].append(index)
    ordered = sorted(i for indices in chosen for i in indices)
    return [None]*(NUM_CAMERAS*per_camera-len(ordered))+[events[i] for i in ordered]


# --------------------------------------------------------------------------- backends

class OnnxGraphs:
    def __init__(self, directory, threads=1, names=GRAPHS):
        import onnxruntime
        options = onnxruntime.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.sessions = {name: onnxruntime.InferenceSession(str(Path(directory)/f'{name}.onnx'), options,
                                                            providers=['CPUExecutionProvider'])
                         for name in names}

    def run(self, name, *arrays):
        session = self.sessions[name]
        feed = {value.name: np.asarray(array, np.float32)[None] for value, array in zip(session.get_inputs(), arrays)}
        return session.run(None, feed)[0][0]


class NcnnGraphs:
    """ncnn nets. Inputs are copied into ncnn-owned Mats and kept alive until extract():
    the Python binding crashes when an in-place layer writes into numpy-backed memory."""

    def __init__(self, directory, threads=1, fp16=True, names=GRAPHS):
        import ncnn
        self.ncnn = ncnn
        self.nets = {}
        for name in names:
            net = ncnn.Net()
            net.opt.use_vulkan_compute = False
            for option in ('use_fp16_packed', 'use_fp16_storage', 'use_fp16_arithmetic'):
                setattr(net.opt, option, fp16)
            if threads:
                net.opt.num_threads = threads
            prefix = str(Path(directory)/name)
            if net.load_param(prefix+'.ncnn.param') or net.load_model(prefix+'.ncnn.bin'):
                raise RuntimeError(f'Cannot load ncnn graph {prefix}')
            self.nets[name] = net

    def run(self, name, *arrays):
        # Keep the float32 copies alive until cloned: a Mat only borrows its array's memory, so a
        # temporary freed after Mat() made clone() copy freed memory (garbage inputs from float64).
        arrays = [np.ascontiguousarray(array, np.float32) for array in arrays]
        mats = [self.ncnn.Mat(array).clone() for array in arrays]
        extractor = self.nets[name].create_extractor()
        for i, mat in enumerate(mats):
            extractor.input(f'in{i}', mat)
        status, output = extractor.extract('out0')
        if status:
            raise RuntimeError(f'ncnn {name} failed with {status}')
        return np.array(output).copy()


# --------------------------------------------------------------------------- runtime

@dataclass
class _Event:
    camera: int
    capture: float
    arrival: float
    origin: np.ndarray     # [3,2,21,3] each camera's latest ray at this arrival
    direction: np.ndarray
    mask: np.ndarray       # [3,2,21]
    ray_capture: np.ndarray  # [3]
    evident: bool
    point: np.ndarray      # [2,21,3] nominal triangulation of those rays
    ok: np.ndarray         # [2,21]
    stamp: np.ndarray      # [2,21] mean capture time of the rays used
    tokens: np.ndarray     # [event_tokens,dim]
    raw: np.ndarray        # [2,21,11], invalid joints zeroed
    valid: np.ndarray      # [2,21]


class LiteRuntime:
    def __init__(self, directory, backend='ncnn', threads=1, fp16=True, graphs=None):
        """graphs: any object with run(name, *arrays), instead of loading a backend."""
        directory = Path(directory)
        meta = json.loads((directory/META_FILE).read_text(encoding='utf-8'))
        if meta.get('architecture') != 'lite':
            raise ValueError(f'{directory} is not a HandLite export')
        if meta.get('export_format') not in READABLE_FORMATS:
            raise ValueError(f'{directory} was exported for another runtime version; re-run python -m training.export')
        self.config = saved_config('lite', meta['config'])
        if graphs is not None:
            self.graphs = graphs
        elif backend == 'ncnn':
            self.graphs = NcnnGraphs(directory, threads, fp16)
        elif backend == 'onnxruntime':
            self.graphs = OnnxGraphs(directory, threads)
        else:
            raise ValueError("backend must be 'ncnn' or 'onnxruntime'")
        self.keep_s = self.config.context_s+STREAM_KEEP_MARGIN_S
        self.reset()

    def reset(self):
        self.events = []
        self.latest = [None]*NUM_CAMERAS          # newest accepted event of each camera
        self.last_arrival = None                  # kept even when every event is pruned

    def __len__(self):
        return len(self.events)

    def push(self, camera, features, valid, capture_time, arrival_time):
        """Encode one arrived camera frame; False (ignored) for a capture no newer than
        that camera's latest. Frames must arrive in order; frames without detections count."""
        camera = check_event(camera, np.shape(features), np.shape(valid))
        capture, arrival = float(capture_time), float(arrival_time)
        if self.last_arrival is not None and arrival < self.last_arrival:
            raise ValueError('Events must be pushed in arrival order')
        self.last_arrival = arrival
        if self.latest[camera] is not None and capture <= self.latest[camera].capture:
            return False
        valid = np.array(valid, dtype=bool, copy=True)      # stored: callers may reuse their buffer
        raw = np.where(valid[..., None], np.asarray(features, np.float64)[..., :MODEL_CHANNELS], 0.)
        origin = np.zeros((NUM_CAMERAS, NUM_HANDS, NUM_JOINTS, 3))
        direction = np.zeros_like(origin)
        mask = np.zeros((NUM_CAMERAS, NUM_HANDS, NUM_JOINTS), bool)
        ray_capture = np.zeros(NUM_CAMERAS)
        newest = list(self.latest)
        newest[camera] = _Event(camera, capture, arrival, None, None, None, None, False, None, None, None, None, raw, valid)
        for c, event in enumerate(newest):
            if event is not None and event.capture >= arrival-self.config.sample_max_age_s:
                origin[c], direction[c] = event.raw[..., RAY_ORIGIN], event.raw[..., RAY_DIRECTION]
                mask[c], ray_capture[c] = event.valid, event.capture
        point, ok, stamp = sample(origin, direction, mask, ray_capture)
        features = joint_features(raw, valid, point, ok).reshape(NUM_HANDS, -1)
        tokens = self.graphs.run('encoder', features, np.eye(NUM_CAMERAS)[camera])
        event = _Event(camera, capture, arrival, origin, direction, mask, ray_capture, bool((ok & valid).any()),
                       point, ok, stamp, tokens, raw, valid)
        self.latest[camera] = event
        self.events.append(event)
        while self.events and self.events[0].capture < arrival-self.keep_s:
            self.events.pop(0)
        return True

    def slots(self, time):
        """See select_slot_events."""
        return select_slot_events(self.events, time, self.config.slots_per_camera, self.config.event_span_s)

    def _single_rays(self, latest, params, time):
        """anchor_at's single-ray inputs (see HandLite.single_rays): each camera's corrected
        ray of each joint in the latest slot event's table, and the joint's newest nominal
        triangulation captured within anchor_depth_memory_s."""
        origin, direction = calibrate(latest.origin, latest.direction, params)
        joints_last = lambda value: np.moveaxis(value, 0, 2)                                    # [2,21,3(,3)]
        hands, joints = np.indices((NUM_HANDS, NUM_JOINTS))
        stored = [e for e in self.events if e.arrival <= time+ARRIVAL_TOLERANCE_S]
        ok = np.stack([e.ok for e in stored]) & (np.stack([e.stamp for e in stored]) >= time-self.config.anchor_depth_memory_s)
        last = np.where(ok, np.arange(len(stored))[:, None, None], -1).max(0)
        reference = np.stack([e.point for e in stored])[np.maximum(last, 0), hands, joints]
        return (joints_last(origin), joints_last(direction), joints_last(latest.mask),
                np.broadcast_to(latest.ray_capture, (NUM_HANDS, NUM_JOINTS, NUM_CAMERAS)), reference, last >= 0)

    def query(self, time):
        """Pose [2,21,3] at time (not before the latest arrival)."""
        pose, _ = self.query_details(time)
        return pose

    def query_details(self, time):
        """Pose [2,21,3] and the camera correction used [3,6]."""
        time = float(time)
        if self.last_arrival is None:
            raise ValueError('No events yet')
        if time < self.last_arrival:
            raise ValueError('Query before the latest arrival')
        slots = self.slots(time)
        dim, per = self.config.dim, self.config.event_tokens
        present = np.array([event is not None for event in slots])
        tokens = np.stack([event.tokens if event is not None else np.zeros((per, dim)) for event in slots])
        tokens = tokens.reshape(-1, dim)
        cameras = np.array([event.camera if event is not None else -1 for event in slots])
        token_valid = np.repeat(present, per)
        token_camera = np.repeat(cameras, per)
        params = np.zeros((NUM_CAMERAS, 6))
        if self.config.calibration_head:
            pool = np.where((token_camera[None] == np.arange(NUM_CAMERAS)[:, None]) & token_valid[None], 0., MASK_OFF)
            evident = np.array([event is not None and event.evident for event in slots])
            informative = ((cameras[None] == np.arange(NUM_CAMERAS)[:, None]) & evident[None]).any(-1)
            params = np.where(informative[:, None], self.graphs.run('calibrator', tokens, pool), 0.)
        used = [event for event in slots if event is not None]
        if used:
            point, ok, stamp = sample(np.stack([e.origin for e in used]), np.stack([e.direction for e in used]),
                                      np.stack([e.mask for e in used]), np.stack([e.ray_capture for e in used]), params,
                                      self.config.ray_outlier_ratio)
        else:
            point = np.zeros((0, NUM_HANDS, NUM_JOINTS, 3))
            ok = np.zeros((0, NUM_HANDS, NUM_JOINTS), bool)
            stamp = np.zeros((0, NUM_HANDS, NUM_JOINTS))
        single = self._single_rays(used[-1], params, time) if used and self.config.anchor_ray_depth else None
        anchor, flags = anchor_at(point, ok, stamp, time, self.config.anchor_lookback_s, self.config.anchor_hold_s,
                                  single)
        # Without any slot there is no ray: the single-ray flag is off, as in torch.
        flags = np.pad(flags, ((0, 0), (0, 0), (0, self.config.anchor_features-3-flags.shape[-1])))
        capture = np.array([event.capture if event is not None else time for event in slots])
        gaps = np.repeat(np.where(present, np.maximum(time-capture, 0.)/TIME_UNIT_S, 0.), per)
        features = np.concatenate((anchor, flags), -1).reshape(HAND_JOINTS, -1)
        offset = self.graphs.run('decoder', tokens, token_valid, gaps, features)
        return anchor+offset.reshape(NUM_HANDS, NUM_JOINTS, 3), params
