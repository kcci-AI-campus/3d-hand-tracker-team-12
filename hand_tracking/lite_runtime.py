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
from .config import LiteConfig
from .constants import (NUM_CAMERAS, NUM_HANDS, NUM_JOINTS, HAND_JOINTS, MODEL_CHANNELS, UV, RAY_ORIGIN,
                        RAY_DIRECTION, DELAY, TIME_UNIT_S, MISS_SCALE, ARRIVAL_TOLERANCE_S, REPEAT_TOLERANCE_S,
                        MIN_TIME_STD_S, MAX_EXTRAPOLATION_S, MIN_RAY_SIN2)

MASK_OFF = -1e4                 # hand_tracking.lite.MASK_OFF
STREAM_KEEP_MARGIN_S = .5       # hand_tracking.stream.STREAM_KEEP_MARGIN_S
META_FILE = 'lite.json'
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


def sample(origins, directions, mask, capture, params=None):
    """Rays [...,3,2,21,3] of the three cameras (optionally corrected by params [3,6]) ->
    point [...,2,21,3], ok [...,2,21], stamp (mean capture of the used rays) [...,2,21]."""
    if params is not None:
        rotation = rodrigues(params[:, :3])                                        # [3,3,3]
        origins = origins+params[:, None, None, 3:]
        directions = np.einsum('cxy,...chjy->...chjx', rotation, directions)
    # [...,camera,hand,joint,xyz] -> [...,hand,joint,camera,xyz]
    rays = list(range(origins.ndim-4))+[origins.ndim-3, origins.ndim-2, origins.ndim-4, origins.ndim-1]
    masks = list(range(mask.ndim-3))+[mask.ndim-2, mask.ndim-1, mask.ndim-3]
    point, ok = triangulate(np.transpose(origins, rays), np.transpose(directions, rays), np.transpose(mask, masks))
    weight = np.transpose(mask, masks).astype(np.float64)
    stamp = (capture[..., None, None, :]*weight).sum(-1)/np.maximum(weight.sum(-1), 1)
    return point, ok, stamp


def anchor_at(point, ok, stamp, query, lookback_s, hold_s):
    """Anchor [2,21,3] and flags [2,21,5] at query from slot samples [S,...] in arrival
    order (all arrived by the query); see events.query_anchor."""
    s = len(point)
    index = np.arange(s)
    now = ok[-1] if s else np.zeros(ok.shape[1:], bool)
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
    weight = joint[..., None].astype(np.float64)
    hand_mean = (anchor*weight).sum(-2, keepdims=True)/np.maximum(weight.sum(-2, keepdims=True), 1)
    hand = np.broadcast_to(joint.any(-1, keepdims=True), joint.shape)
    anchor = np.where(joint[..., None], anchor, hand_mean)
    flags = np.stack((now, joint, hand, age/TIME_UNIT_S, moving), -1).astype(np.float64)
    return anchor, flags


def joint_features(raw, valid, point, ok):
    """Event features [2,21,11], valid, nominal sample -> encoder input [2,21,14]."""
    origin, direction = raw[..., RAY_ORIGIN], raw[..., RAY_DIRECTION]
    relative = point-origin
    miss = relative-direction*(relative*direction).sum(-1, keepdims=True)
    miss = np.where((valid & ok)[..., None], miss*MISS_SCALE, 0.)
    x = np.concatenate((raw[..., UV]*2-1, origin, direction, raw[..., DELAY]/TIME_UNIT_S, miss,
                        valid[..., None], (ok & valid)[..., None]), -1)
    return np.where(valid[..., None], x, 0.)


# --------------------------------------------------------------------------- backends

class OnnxGraphs:
    def __init__(self, directory, threads=1):
        import onnxruntime
        options = onnxruntime.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.sessions = {name: onnxruntime.InferenceSession(str(Path(directory)/f'{name}.onnx'), options,
                                                            providers=['CPUExecutionProvider'])
                         for name in GRAPHS}

    def run(self, name, *arrays):
        session = self.sessions[name]
        feed = {value.name: np.asarray(array, np.float32)[None] for value, array in zip(session.get_inputs(), arrays)}
        return session.run(None, feed)[0][0]


class NcnnGraphs:
    """ncnn nets. Inputs are copied into ncnn-owned Mats and kept alive until extract():
    the Python binding crashes when an in-place layer writes into numpy-backed memory."""

    def __init__(self, directory, threads=1, fp16=True):
        import ncnn
        self.ncnn = ncnn
        self.nets = {}
        for name in GRAPHS:
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
        mats = [self.ncnn.Mat(np.ascontiguousarray(array, np.float32)).clone() for array in arrays]
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
    tokens: np.ndarray     # [2,dim]
    raw: np.ndarray        # [2,21,11], invalid joints zeroed
    valid: np.ndarray      # [2,21]


class LiteRuntime:
    def __init__(self, directory, backend='ncnn', threads=1, fp16=True):
        directory = Path(directory)
        meta = json.loads((directory/META_FILE).read_text(encoding='utf-8'))
        if meta.get('architecture') != 'lite':
            raise ValueError(f'{directory} is not a HandLite export')
        self.config = LiteConfig(**meta['config'])
        if backend == 'ncnn':
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

    def __len__(self):
        return len(self.events)

    def push(self, camera, features, valid, capture_time, arrival_time):
        """Encode one arrived camera frame; False (ignored) for a capture no newer than
        that camera's latest. Frames must arrive in order; frames without detections count."""
        camera, capture, arrival = int(camera), float(capture_time), float(arrival_time)
        if self.events and arrival < self.events[-1].arrival:
            raise ValueError('Events must be pushed in arrival order')
        if self.latest[camera] is not None and capture <= self.latest[camera].capture:
            return False
        valid = np.asarray(valid, bool)
        raw = np.where(valid[..., None], np.asarray(features, np.float64)[..., :MODEL_CHANNELS], 0.)
        origin = np.zeros((NUM_CAMERAS, NUM_HANDS, NUM_JOINTS, 3))
        direction = np.zeros_like(origin)
        mask = np.zeros((NUM_CAMERAS, NUM_HANDS, NUM_JOINTS), bool)
        ray_capture = np.zeros(NUM_CAMERAS)
        newest = list(self.latest)
        newest[camera] = _Event(camera, capture, arrival, None, None, None, None, False, None, raw, valid)
        for c, event in enumerate(newest):
            if event is not None and event.capture >= arrival-self.config.sample_max_age_s:
                origin[c], direction[c] = event.raw[..., RAY_ORIGIN], event.raw[..., RAY_DIRECTION]
                mask[c], ray_capture[c] = event.valid, event.capture
        point, ok, _ = sample(origin, direction, mask, ray_capture)
        features = joint_features(raw, valid, point, ok).reshape(NUM_HANDS, -1)
        tokens = self.graphs.run('encoder', features, np.eye(NUM_CAMERAS)[camera])
        event = _Event(camera, capture, arrival, origin, direction, mask, ray_capture, bool((ok & valid).any()),
                       tokens, raw, valid)
        self.latest[camera] = event
        self.events.append(event)
        while self.events and self.events[0].capture < arrival-self.keep_s:
            self.events.pop(0)
        return True

    def slots(self, time):
        """Each camera's K latest events arrived by time and captured within the span, in
        arrival order; None for padding (first)."""
        k, span = self.config.slots_per_camera, self.config.event_span_s
        per_camera = [[] for _ in range(NUM_CAMERAS)]
        for index in range(len(self.events)-1, -1, -1):
            event = self.events[index]
            if event.arrival <= time+ARRIVAL_TOLERANCE_S and event.capture >= time-span \
                    and len(per_camera[event.camera]) < k:
                per_camera[event.camera].append(index)
        chosen = sorted(i for indices in per_camera for i in indices)
        return [None]*(NUM_CAMERAS*k-len(chosen))+[self.events[i] for i in chosen]

    def query(self, time):
        """Pose [2,21,3] at time (not before the latest arrival)."""
        time = float(time)
        if not self.events:
            raise ValueError('No events yet')
        if time < self.events[-1].arrival:
            raise ValueError('Query before the latest arrival')
        pose, _ = self.query_details(time)
        return pose

    def query_details(self, time):
        """Pose [2,21,3] and the camera correction used [3,6]."""
        slots = self.slots(time)
        dim = self.config.dim
        present = np.array([event is not None for event in slots])
        tokens = np.stack([event.tokens if event is not None else np.zeros((NUM_HANDS, dim)) for event in slots])
        tokens = tokens.reshape(-1, dim)
        cameras = np.array([event.camera if event is not None else -1 for event in slots])
        token_valid = np.repeat(present, NUM_HANDS)
        token_camera = np.repeat(cameras, NUM_HANDS)
        params = np.zeros((NUM_CAMERAS, 6))
        if self.config.calibration_head:
            pool = np.where((token_camera[None] == np.arange(NUM_CAMERAS)[:, None]) & token_valid[None], 0., MASK_OFF)
            evident = np.array([event is not None and event.evident for event in slots])
            informative = ((cameras[None] == np.arange(NUM_CAMERAS)[:, None]) & evident[None]).any(-1)
            params = np.where(informative[:, None], self.graphs.run('calibrator', tokens, pool), 0.)
        used = [event for event in slots if event is not None]
        if used:
            point, ok, stamp = sample(np.stack([e.origin for e in used]), np.stack([e.direction for e in used]),
                                      np.stack([e.mask for e in used]), np.stack([e.ray_capture for e in used]), params)
        else:
            point = np.zeros((0, NUM_HANDS, NUM_JOINTS, 3))
            ok = np.zeros((0, NUM_HANDS, NUM_JOINTS), bool)
            stamp = np.zeros((0, NUM_HANDS, NUM_JOINTS))
        anchor, flags = anchor_at(point, ok, stamp, time, self.config.anchor_lookback_s, self.config.anchor_hold_s)
        capture = np.array([event.capture if event is not None else time for event in slots])
        gaps = np.repeat(np.maximum(time-capture, 0.)/TIME_UNIT_S, NUM_HANDS)
        features = np.concatenate((anchor, flags), -1).reshape(HAND_JOINTS, -1)
        offset = self.graphs.run('decoder', tokens, np.where(token_valid, 0., MASK_OFF), gaps, features)
        return anchor+offset.reshape(NUM_HANDS, NUM_JOINTS, 3), params
