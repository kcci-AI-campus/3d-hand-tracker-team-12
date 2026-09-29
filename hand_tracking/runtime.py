"""HandLiteV3 deployment runtime without PyTorch: the geometry (line-fit rays, triangulation,
anchors, joint features) and slot selection in numpy around the exported ncnn networks (encoder
per event, corrector per query). Mirrors hand_tracking.model.LiteV3Stream; no state besides the
recent events and each joint's last triangulation (the state: a joint seen by one camera keeps its
depth however long).

    runtime = LiteV3Runtime('deploy_dir')
    runtime.push(camera, features [2,21,14], valid [2,21], capture_time, arrival_time)
    pose = runtime.query(time)          # [2,21,3] world units
    runtime.error_mm                    # [2,21] each joint's expected error in mm (or None)
    runtime.anchor_kind                 # [2,21] how each joint was anchored (model.ANCHOR_KINDS index)
    runtime.in_view_probability         # [2] each hand's probability of being inside some camera's view
                                        #     (below 0.5: out of all cameras, ignore that hand's joints)

Times are absolute seconds (float64). Needs numpy and `ncnn`.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from .config import saved_config
from .constants import (NUM_CAMERAS, NUM_HANDS, NUM_JOINTS, HAND_JOINTS, FINGERS, MODEL_CHANNELS, UV, RAY_ORIGIN,
                        RAY_DIRECTION, DELAY, TIME_UNIT_S, MISS_SCALE, RESIDUAL_UNIT, STREAM_KEEP_MARGIN_S,
                        ARRIVAL_TOLERANCE_S, MIN_TIME_STD_S, MIN_RAY_SIN2, check_event)

META_FILE = 'litev3.json'
# 2: failed joints anchored from their own last triangulation (prior_span_s, prior_step_s); 3: plus the state
# (each joint's last triangulation however old), prior age capped at max_prior_age_s; 4: each hand's in-view
# logit as the corrector's last column (config.presence)
EXPORT_FORMAT = 4
GRAPHS = ('encoder', 'corrector')
EVENT_TOKENS = NUM_HANDS*FINGERS     # finger tokens per event
FAR = 1e6


def direct_features(raw, valid):
    """Event features [...,2,21,11] (invalid joints zeroed), valid -> encoder input [...,2,21,10]
    (networks.direct_features)."""
    x = np.concatenate((raw[..., UV]*2-1, raw[..., RAY_ORIGIN], raw[..., RAY_DIRECTION], raw[..., DELAY]/TIME_UNIT_S,
                        valid[..., None]), -1)
    return np.where(valid[..., None], x, 0.)


def triangulate(origins, directions, mask):
    """origins/directions [...,C,3], mask [...,C] -> point [...,3], valid [...] (geometry.triangulate)."""
    mask = (mask & np.isfinite(origins).all(-1) & np.isfinite(directions).all(-1)
            & (np.linalg.norm(directions, axis=-1) > 1e-8))
    d = np.where(mask[..., None], directions, 0.)
    o = np.where(mask[..., None], origins, 0.)
    d = d/np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-8)
    count = mask.sum(-1)
    a = count[..., None, None]*np.eye(3)-np.einsum('...ci,...cj->...ij', d, d)
    rhs = (o-d*(d*o).sum(-1, keepdims=True)).sum(-2)
    sin2 = np.zeros(count.shape)
    for i in range(d.shape[-2]):
        for j in range(i+1, d.shape[-2]):
            sin2 = np.maximum(sin2, np.square(np.cross(d[..., i, :], d[..., j, :])).sum(-1))
    valid = (count >= 2) & (sin2 > MIN_RAY_SIN2)
    point = np.linalg.solve(np.where(valid[..., None, None], a, np.eye(3)), rhs[..., None])[..., 0]
    depth = ((point[..., None, :]-o)*d).sum(-1)
    valid = valid & ((depth > 0) | ~mask).all(-1)
    return np.where(valid[..., None], point, 0.), valid


def ray_residuals(point, origins, directions):
    """Angular residual of each ray [...,C] to point [...,3] (geometry.ray_residuals)."""
    d = directions/np.maximum(np.linalg.norm(directions, axis=-1, keepdims=True), 1e-8)
    relative = point[..., None, :]-origins
    depth = (relative*d).sum(-1)
    return np.linalg.norm(relative-depth[..., None]*d, axis=-1)/np.maximum(depth, 1e-6)


def unit(d):
    return d/np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-8)


def fitted_rays(events, time, span_s, offset_s=0.):
    """model.fitted_rays for one query at time - offset_s: per camera and joint [3,42,...]."""
    c, j = NUM_CAMERAS, HAND_JOINTS
    reference = time-offset_s
    origin, direction, newest_direction = np.zeros((c, j, 3)), np.zeros((c, j, 3)), np.zeros((c, j, 3))
    has, fit = np.zeros((c, j), bool), np.zeros((c, j), bool)
    count, age, oldest, rms, residual = (np.zeros((c, j)) for _ in range(5))
    uv = np.zeros((c, j, 2))
    for camera in range(c):
        usable = [e for e in events if e.camera == camera and e.arrival <= time+ARRIVAL_TOLERANCE_S
                  and reference-span_s <= e.capture <= reference+ARRIVAL_TOLERANCE_S]
        if not usable:
            continue
        use = np.stack([e.valid.reshape(j) for e in usable])                               # [n,42]
        m = use.astype(np.float64)
        tau = np.array([e.capture-reference for e in usable])[:, None]                    # [n,1]
        raw = np.stack([e.raw.reshape(j, -1) for e in usable])                             # [n,42,11]
        d, o = raw[..., RAY_DIRECTION], raw[..., RAY_ORIGIN]
        mt = m*tau
        s0, s1, s2 = m.sum(0), mt.sum(0), (mt*tau).sum(0)
        sd, sdt = (m[..., None]*d).sum(0), (mt[..., None]*d).sum(0)
        spread = s0*s2-s1*s1
        fitted = (s0 >= 2) & (spread > MIN_TIME_STD_S**2*s0*s0)
        slope = np.where(fitted[:, None], (s0[:, None]*sdt-s1[:, None]*sd)/np.maximum(spread, 1e-12)[:, None], 0.)
        value = (sd-slope*s1[:, None])/np.maximum(s0, 1)[:, None]                         # the line at the query time
        squared = np.maximum((m*np.square(d).sum(-1)).sum(0)-2*(value*sd).sum(-1)-2*(slope*sdt).sum(-1)
                             +np.square(value).sum(-1)*s0+2*(value*slope).sum(-1)*s1+np.square(slope).sum(-1)*s2, 0.)
        scale = np.maximum(np.linalg.norm(value, axis=-1), 1e-8)
        seen = use.any(0)
        newest_index = np.where(use, np.arange(len(usable))[:, None], -1).max(0).clip(0)   # [42]
        newest = raw[newest_index, np.arange(j)]                                          # [42,11]
        newest_tau, oldest_tau = np.where(use, tau, -FAR).max(0), np.where(use, tau, FAR).min(0)
        newest_residual = np.linalg.norm(newest[:, RAY_DIRECTION]-value-slope*np.maximum(newest_tau, -span_s)[:, None],
                                         axis=-1)/scale
        origin[camera] = (m[..., None]*o).sum(0)/np.maximum(s0, 1)[:, None]
        direction[camera], has[camera], fit[camera], count[camera] = unit(value), seen, fitted, s0
        age[camera], oldest[camera] = np.where(seen, -newest_tau, 0.), np.where(seen, -oldest_tau, 0.)
        uv[camera], newest_direction[camera] = newest[:, UV], unit(newest[:, RAY_DIRECTION])
        rms[camera] = np.where(seen, np.sqrt(squared/np.maximum(s0, 1))/scale, 0.)
        residual[camera] = np.where(seen, newest_residual, 0.)
    return origin, direction, has, fit, count, age, oldest, uv, newest_direction, rms, residual


def rays_at(events, time, config, offset_s=0.):
    """model.rays_at for one query: fitted_rays per joint and camera [2,21,3,...]."""
    return tuple(value.swapaxes(0, 1).reshape(NUM_HANDS, NUM_JOINTS, NUM_CAMERAS, *value.shape[2:])
                 for value in fitted_rays(events, time, config.fit_span_s, offset_s))


def prior_points(events, time, config):
    """model.prior_points for one query -> point [2,21,3], found [2,21], age (s) [2,21]."""
    point, found, age = np.zeros((NUM_HANDS, NUM_JOINTS, 3)), np.zeros((NUM_HANDS, NUM_JOINTS), bool), \
        np.zeros((NUM_HANDS, NUM_JOINTS))
    for k in range(1, config.prior_steps+1):
        offset = k*config.prior_step_s
        origin, direction, has, _, _, newest_age = rays_at(events, time, config, offset)[:6]
        p, ok = triangulate(origin, direction, has)
        take = ok & ~found
        point = np.where(take[..., None], p, point)
        age = np.where(take, np.where(has, newest_age, FAR).min(-1)+offset, age)
        found = found | ok
    return point, found, age


def anchor_features(events, time, config, stored=None):
    """model.anchor_features for one query -> anchor [2,21,3], kind [2,21], joint features [42,F]
    and where triangulated the age of the newest detection used (s) [2,21]. stored: the state's
    (point [2,21,3], known [2,21], age (s) [2,21]), used only where the history search finds none."""
    origin, direction, has, fit, count, age, oldest, uv, newest_direction, rms, residual = rays_at(events, time, config)
    point, ok = triangulate(origin, direction, has)
    prior, found, prior_age = prior_points(events, time, config)
    if stored is not None:
        kept, known, kept_age = stored
        use = known & ~found
        prior, found, prior_age = np.where(use[..., None], kept, prior), found | known, np.where(use, kept_age, prior_age)
    observed = np.where(has, age, FAR).min(-1)
    prior_age = np.where(found, np.minimum(prior_age, config.max_prior_age_s), 0.)
    miss = np.where(has, ray_residuals(prior, origin, direction), FAR)                      # [2,21,3]
    best = miss.argmin(-1)
    o = np.take_along_axis(origin, best[..., None, None], -2)[..., 0, :]
    dd = np.take_along_axis(direction, best[..., None, None], -2)[..., 0, :]
    projected = o+np.maximum(((prior-o)*dd).sum(-1, keepdims=True), 0.)*dd
    seen = has.any(-1)
    on_ray, at_prior = ~ok & found & seen, ~ok & found & ~seen
    anchor = np.where(ok[..., None], point, np.where(on_ray[..., None], projected,
                                                     np.where(at_prior[..., None], prior, 0.)))
    kind = ok*1+on_ray*2+at_prior*3
    relative = anchor[..., None, :]-origin
    miss_vector = relative-direction*(relative*direction).sum(-1, keepdims=True)
    camera = np.concatenate((has[..., None], fit[..., None], count[..., None]/4, age[..., None]/TIME_UNIT_S,
                             oldest[..., None]/TIME_UNIT_S, uv*2-1, miss_vector*MISS_SCALE,
                             (direction-newest_direction)*MISS_SCALE, rms[..., None]/RESIDUAL_UNIT,
                             residual[..., None]/RESIDUAL_UNIT), -1)
    camera = np.where(has[..., None], camera, 0.)                                           # [2,21,3,15]
    joint = np.concatenate((anchor, np.where(found[..., None], anchor-prior, 0.), ok[..., None], on_ray[..., None],
                            at_prior[..., None], found[..., None], prior_age[..., None]/TIME_UNIT_S), -1)
    features = np.concatenate((joint, camera.reshape(NUM_HANDS, NUM_JOINTS, -1)), -1).astype(np.float64)
    return anchor, kind, features.reshape(HAND_JOINTS, -1), np.where(ok, observed, 0.)


def select_slot_events(events, time, per_camera, span_s):
    """Each camera's per_camera latest events (with .camera/.capture/.arrival, in arrival
    order) arrived by time and captured within span_s, in arrival order; None for padding
    (first). events.select_slots for one query."""
    chosen = [[] for _ in range(NUM_CAMERAS)]
    for index in range(len(events)-1, -1, -1):
        event = events[index]
        if event.arrival <= time+ARRIVAL_TOLERANCE_S and event.capture >= time-span_s \
                and len(chosen[event.camera]) < per_camera:
            chosen[event.camera].append(index)
    ordered = sorted(i for indices in chosen for i in indices)
    return [None]*(NUM_CAMERAS*per_camera-len(ordered))+[events[i] for i in ordered]


class NcnnGraphs:
    """ncnn nets. Inputs are copied into ncnn-owned Mats and kept alive until extract():
    the Python binding crashes when an in-place layer writes into numpy-backed memory."""

    def __init__(self, directory, names, threads=1, fp16=True):
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
        # Keep the float32 copies alive until cloned: a Mat only borrows its array's memory.
        arrays = [np.ascontiguousarray(array, np.float32) for array in arrays]
        mats = [self.ncnn.Mat(array).clone() for array in arrays]
        extractor = self.nets[name].create_extractor()
        for i, mat in enumerate(mats):
            extractor.input(f'in{i}', mat)
        status, output = extractor.extract('out0')
        if status:
            raise RuntimeError(f'ncnn {name} failed with {status}')
        return np.array(output).copy()


@dataclass
class _Event:
    camera: int
    capture: float
    arrival: float
    raw: np.ndarray      # [2,21,11], undetected joints zeroed
    valid: np.ndarray    # [2,21]
    tokens: np.ndarray   # [10,dim] finger tokens


class LiteV3Runtime:
    def __init__(self, directory, threads=1, fp16=True, graphs=None):
        """graphs: any object with run(name, *arrays), instead of loading ncnn."""
        directory = Path(directory)
        meta = json.loads((directory/META_FILE).read_text(encoding='utf-8'))
        if meta.get('export_format') != EXPORT_FORMAT:
            raise ValueError(f'{directory} is not a HandLiteV3 export of format {EXPORT_FORMAT}')
        self.config = c = saved_config(meta['config'])
        self.graphs = graphs if graphs is not None else NcnnGraphs(directory, GRAPHS, threads, fp16)
        self.keep_s = c.context_s+STREAM_KEEP_MARGIN_S
        self.reset()

    def reset(self):
        self.events = []
        self.newest_capture = [None]*NUM_CAMERAS
        self.last_arrival = None
        self.error_mm = None
        self.anchor_kind = None
        self.in_view_probability = None   # [2] each hand's probability of being inside some camera's view
        # State: each joint's last triangulation and the capture time of the newest detection it used.
        self.prior_point = np.zeros((NUM_HANDS, NUM_JOINTS, 3))
        self.prior_capture = np.zeros((NUM_HANDS, NUM_JOINTS))
        self.prior_known = np.zeros((NUM_HANDS, NUM_JOINTS), bool)

    def __len__(self):
        return len(self.events)

    def push(self, camera, features, valid, capture_time, arrival_time):
        """Encode one arrived camera frame; False (ignored) for a capture no newer than that
        camera's latest. Frames must arrive in order; frames without detections count."""
        camera = check_event(camera, np.shape(features), np.shape(valid))
        capture, arrival = float(capture_time), float(arrival_time)
        if self.last_arrival is not None and arrival < self.last_arrival:
            raise ValueError('Events must be pushed in arrival order')
        self.last_arrival = arrival
        if self.newest_capture[camera] is not None and capture <= self.newest_capture[camera]:
            return False
        self.newest_capture[camera] = capture
        valid = np.array(valid, dtype=bool, copy=True)
        raw = np.where(valid[..., None], np.asarray(features, np.float64)[..., :MODEL_CHANNELS], 0.)
        tokens = np.asarray(self.graphs.run('encoder', direct_features(raw, valid).reshape(NUM_HANDS, -1),
                                            np.eye(NUM_CAMERAS)[camera]))
        self.events.append(_Event(camera, capture, arrival, raw, valid, tokens))
        while self.events and self.events[0].capture < arrival-self.keep_s:
            self.events.pop(0)
        return True

    def query(self, time):
        """Pose [2,21,3] at time (not before the latest arrival)."""
        time = float(time)
        if self.last_arrival is None:
            raise ValueError('No events yet')
        if time < self.last_arrival:
            raise ValueError('Query before the latest arrival')
        c = self.config
        stored = (self.prior_point, self.prior_known, time-self.prior_capture)
        anchor, self.anchor_kind, features, observed = anchor_features(self.events, time, c, stored)
        triangulated = self.anchor_kind == 1
        self.prior_point = np.where(triangulated[..., None], anchor, self.prior_point)
        self.prior_capture = np.where(triangulated, time-observed, self.prior_capture)
        self.prior_known = self.prior_known | triangulated
        slots = select_slot_events(self.events, time, c.slots_per_camera, c.event_span_s)
        present = np.array([event is not None for event in slots])
        tokens = np.stack([event.tokens if event is not None else np.zeros((EVENT_TOKENS, c.dim)) for event in slots])
        ages = np.array([max(time-event.capture, 0.)/TIME_UNIT_S if event is not None else 0. for event in slots])
        output = np.asarray(self.graphs.run('corrector', features, tokens.reshape(-1, c.dim),
                                            np.repeat(present, EVENT_TOKENS).astype(np.float64),
                                            np.repeat(ages, EVENT_TOKENS))).reshape(HAND_JOINTS, -1)
        column = 3
        if c.error_estimate:                                 # next column: expected error, log(1 + mm)
            self.error_mm = np.expm1(output[:, column]).reshape(NUM_HANDS, NUM_JOINTS)
            column += 1
        if c.presence:                                       # next column: the hand's in-view logit (on its joints)
            logits = output[:, column].reshape(NUM_HANDS, NUM_JOINTS)[:, 0]
            self.in_view_probability = 1/(1+np.exp(-logits))
        return anchor+output[:, :3].reshape(NUM_HANDS, NUM_JOINTS, 3)
