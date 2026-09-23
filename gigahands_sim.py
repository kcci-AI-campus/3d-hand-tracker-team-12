"""Reproducible, causal multi-camera hand training data (NumPy only)."""
from dataclasses import dataclass, asdict, field
from pathlib import Path
import argparse
import json
import numpy as np


def default_error_ranges():
    """Installation variation with mild detection noise and measured timing."""
    return {
        'position_std': [.1, .1], 'angle_std_deg': [5, 5],
        'k1': [-.01, .01], 'k2': [-.003, .003],
        'pixel_std': [.15, .6], 'outlier_prob': [0, .001],
        'outlier_std_px': [1, 3], 'missing_prob': [0, .005],
        'packet_loss': [0, .001], 'latency_ms': [1.2, 1.8],
        'latency_jitter_ms': [.5, 1.2], 'processing_ms': [60, 68],
        'capture_jitter_ms': [0, 1], 'clock_offset_std_ms': [0, .3],
        'clock_drift_std_ppm': [0, 2], 'rolling_shutter_ms': [0, 0],
    }


@dataclass
class Config:
    source_fps: float = 30
    camera_fps: float = 17.1
    output_fps: float = 30
    seed: int = 42
    pose_seed: int | None = None
    randomize_errors: bool = True
    error_ranges: dict = field(default_factory=default_error_ranges)
    timing_profile: str = 'profiles/pi_c270_timing.json'
    world_unit_cm: float = 30.0
    position_limit_cm: float | None = None
    angle_limit_deg: float | None = None
    scale: float = 1.0
    center: bool = True
    fit_extent: float = 0.65
    axis_order: str = 'xyz'
    axis_sign: list = field(default_factory=lambda: [1, 1, 1])
    joint_order: list = field(default_factory=lambda: list(range(21)))
    swap_hands: bool = False
    cameras: list = field(default_factory=lambda: [[-1, -1, .5], [1, -1, .5], [0, 1, 1]])
    targets: list = field(default_factory=lambda: [[0, 0, 0]] * 3)
    width: int = 320
    height: int = 240
    diagonal_fov: float = 55
    intrinsics: list = field(default_factory=list)  # optional 3 x [fx,fy,cx,cy]
    position_std: float = .1
    angle_std_deg: float = 5.
    focal_std_pct: float = 0.
    principal_std_px: float = 0.
    k1: float = 0.0
    k2: float = 0.0
    pixel_std: float = .35
    outlier_prob: float = .0005
    outlier_std_px: float = 2.
    missing_prob: float = .002
    packet_loss: float = 0.
    latency_ms: float = 1.47
    latency_jitter_ms: float = .93
    processing_ms: float = 64.33
    capture_jitter_ms: float = .5
    clock_offset_std_ms: float = 0
    clock_drift_std_ppm: float = 0
    rolling_shutter_ms: float = 0
    max_age_ms: float = 250
    window: int = 16
    stride: int = 1

    def validate(self):
        if self.pose_seed is not None and (not isinstance(self.pose_seed, int) or self.pose_seed < 0):
            raise ValueError('pose_seed must be a nonnegative integer or null')
        if not isinstance(self.timing_profile, str):
            raise ValueError('timing_profile must be a path string or empty string')
        if not isinstance(self.randomize_errors, bool) or not isinstance(self.error_ranges, dict):
            raise ValueError('randomize_errors must be boolean; error_ranges must be a JSON object')
        allowed = default_error_ranges()
        for name, bounds in self.error_ranges.items():
            if name not in allowed:
                raise ValueError(f'Unsupported random parameter: {name}')
            a = np.asarray(bounds, float)
            if a.shape != (2,) or not np.isfinite(a).all() or a[0] > a[1]:
                raise ValueError(f'{name}: range must be finite [min,max]')
            if name not in ('k1', 'k2') and a[0] < 0:
                raise ValueError(f'{name}: random range cannot be negative')
            if name in ('outlier_prob', 'missing_prob', 'packet_loss') and a[1] > 1:
                raise ValueError(f'{name}: probability range must be <= 1')
        for name, value in asdict(self).items():
            if isinstance(value, (float, int)) and not np.isfinite(value):
                raise ValueError(f'{name}: finite value required')
        for name in ('source_fps', 'camera_fps', 'output_fps', 'scale', 'max_age_ms', 'world_unit_cm'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        for name in ('width', 'height', 'window', 'stride', 'seed'):
            v = getattr(self, name)
            if not isinstance(v, int) or v < (0 if name == 'seed' else 1):
                raise ValueError(f'{name}: invalid integer')
        for name in ('position_std', 'angle_std_deg', 'focal_std_pct', 'principal_std_px',
                     'pixel_std', 'outlier_std_px', 'latency_ms', 'latency_jitter_ms',
                     'processing_ms', 'capture_jitter_ms', 'clock_offset_std_ms',
                     'clock_drift_std_ppm', 'rolling_shutter_ms', 'position_limit_cm', 'angle_limit_deg'):
            if name in ('position_limit_cm', 'angle_limit_deg') and getattr(self, name) is None:
                continue
            if getattr(self, name) < 0:
                raise ValueError(f'{name} must be nonnegative')
        for name in ('outlier_prob', 'missing_prob', 'packet_loss'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in [0,1]')
        if not 0 <= self.fit_extent < 1 or not 1 < self.diagonal_fov < 170:
            raise ValueError('fit_extent: [0,1), FOV: (1,170)')
        if self.angle_limit_deg is not None and self.angle_limit_deg > 180:
            raise ValueError('angle_limit_deg must be <= 180')
        if sorted(self.axis_order) != list('xyz') or sorted(self.joint_order) != list(range(21)):
            raise ValueError('Invalid axis/joint permutation')
        if len(self.axis_sign) != 3 or any(x not in (-1, 1) for x in self.axis_sign):
            raise ValueError('axis_sign must contain three +/-1 values')
        for name in ('cameras', 'targets'):
            a = np.asarray(getattr(self, name), float)
            if a.shape != (3, 3) or not np.isfinite(a).all():
                raise ValueError(f'{name}: expected finite [3,3]')
        if self.intrinsics:
            k = np.asarray(self.intrinsics, float)
            if k.shape != (3, 4) or not np.isfinite(k).all() or (k[:, :2] <= 0).any():
                raise ValueError('intrinsics: expected 3 x [fx,fy,cx,cy]')


def load_motion(path, fps=30, member=None, *, stream=None):
    """Official sequence JSON arrays; explicit canonical NPY/NPZ also supported."""
    p = Path(path)
    from motion_archive import is_motion_archive, list_motion_members, read_motion_member
    if stream is not None and member is not None:
        raise ValueError('stream and archive member cannot be combined')
    if stream is None and is_motion_archive(path):
        if member is None:
            members = list_motion_members(path)
            if len(members) != 1:
                raise ValueError(f'압축 안에 시퀀스 {len(members)}개가 있습니다. GUI에서 선택하거나 --member를 지정하세요.')
            member = members[0]
        stream = read_motion_member(path, member)
        p = Path(member)
    elif member is not None:
        raise ValueError('--member는 TAR 압축 입력에만 사용할 수 있습니다.')
    times = None
    if p.suffix.lower() == '.npz':
        with np.load(stream if stream is not None else p, allow_pickle=False) as d:
            points = d['points'].copy()
            if 'timestamps' in d:
                times = d['timestamps'].copy()
    elif p.suffix.lower() == '.npy':
        points = np.load(stream if stream is not None else p, allow_pickle=False)
    else:
        data = json.loads(stream.read().decode('utf-8-sig') if stream is not None else p.read_text(encoding='utf-8-sig'))
        if isinstance(data, dict):
            times = data.get('timestamps')
            data = data['points']
        points = np.asarray(data, float)
    points = np.asarray(points, float)
    if points.ndim == 2 and points.shape[1:] == (126,):
        points = points.reshape(-1, 2, 21, 3)
    elif points.ndim == 3 and points.shape[1:] == (42, 3):
        points = points.reshape(-1, 2, 21, 3)
    if points.ndim != 4 or points.shape[1:] != (2, 21, 3) or len(points) < 2:
        raise ValueError('Expected [T,126], [T,42,3], or [T,2,21,3], T >= 2')
    if np.isinf(points).any():
        raise ValueError('Infinite coordinates are invalid; use NaN for missing joints')
    points[~np.isfinite(points).all(axis=-1)] = np.nan
    if not np.isfinite(points).any():
        raise ValueError('No valid joints')
    times = np.arange(len(points)) / fps if times is None else np.asarray(times, float)
    if times.shape != (len(points),) or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError('timestamps must be finite, strictly increasing seconds')
    return points, times - times[0]


def demo_motion(frames=180):
    t = np.arange(frames) / 30
    p = np.zeros((frames, 2, 21, 3))
    for h in range(2):
        p[:, h, 0, 0] = (h * 2 - 1) * .18
        for finger in range(5):
            for joint in range(4):
                j = 1 + finger * 4 + joint
                bend = (1 + np.sin(t * 2 + h)) * .5
                p[:, h, j, 0] = (h * 2 - 1) * .18 + (finger - 2) * .035
                p[:, h, j, 1] = .045 + (joint + 1) * .035 * np.cos(bend)
                p[:, h, j, 2] = (joint + 1) * .035 * np.sin(bend)
        p[:, h, :, 0] += .035 * np.sin(t[:, None] * 1.5)
    return p, t


def normalize(points, cfg):
    p = points[..., ['xyz'.index(a) for a in cfg.axis_order]] * cfg.axis_sign
    p = p[:, ::-1] if cfg.swap_hands else p.copy()
    p = p[:, :, cfg.joint_order]
    center = (np.nanmin(p, axis=(0, 1, 2)) + np.nanmax(p, axis=(0, 1, 2))) / 2 if cfg.center else np.zeros(3)
    p -= center
    scale = cfg.scale
    if cfg.fit_extent:
        extent = np.nanmax(np.abs(p))
        scale = cfg.fit_extent / extent if extent > 0 else 1.
    p *= scale
    if np.nanmax(np.abs(p)) >= 1:
        raise ValueError('Motion exceeds (-1,1). Enable fit_extent or adjust scale/center.')
    return p, {'center_source_axes': center.tolist(), 'effective_scale': scale}


def look_at(position, target):
    forward = np.asarray(target) - position
    if np.linalg.norm(forward) < 1e-8:
        raise ValueError('Camera position and target coincide')
    forward /= np.linalg.norm(forward)
    # Horizon-level camera: right has no world-Y component, hence no roll.
    # atan2(0, 0)=0 fixes an arbitrary but stable heading at an exact pole.
    heading = np.arctan2(forward[0], forward[2])
    right = np.array([-np.cos(heading), 0., np.sin(heading)])
    down = np.cross(forward, right)
    return np.stack([right, down, forward])


def camera_pose_without_roll(rng, nominal, sigma_deg, limit_deg=None):
    """World-Y yaw and elevation pitch Gaussian errors; camera stays level.

    Returns the world-to-camera rotation and sampled [yaw, pitch] offsets.
    Rotation-vector Z=0 alone would not guarantee a level image plane.
    """
    forward = nominal[2]
    yaw = np.arctan2(forward[0], forward[2])
    pitch = np.arctan2(forward[1], np.hypot(forward[0], forward[2]))
    if sigma_deg == 0 or limit_deg == 0:
        return nominal.copy(), np.zeros(2)
    while True:
        offset = bounded_normal_vector(rng, np.deg2rad(sigma_deg),
                                       None if limit_deg is None else np.deg2rad(limit_deg), dimensions=2)
        y, p = np.array([yaw, pitch]) + offset
        direction = np.array([np.sin(y)*np.cos(p), np.sin(p), np.cos(y)*np.cos(p)])
        actual = look_at(np.zeros(3), direction)
        angle = np.rad2deg(np.arccos(np.clip((np.trace(actual @ nominal.T)-1)/2, -1, 1)))
        if limit_deg is None or angle <= limit_deg + 1e-10:
            return actual, np.rad2deg(offset)


def rotation(v):
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = v / angle
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def bounded_normal_vector(rng, sigma, limit, dimensions=3):
    """Isotropic Gaussian conditioned on Euclidean norm <= limit, no clipping."""
    if limit is None:
        return rng.normal(0, sigma, dimensions)
    if sigma == 0 or limit == 0:
        return np.zeros(dimensions)
    while True:
        if sigma <= limit:
            v = rng.normal(0, sigma, dimensions)
            if np.linalg.norm(v) <= limit:
                return v
        else:
            # Uniform-ball proposal avoids slow rejection for tiny limits.
            direction = rng.normal(size=dimensions)
            length = np.linalg.norm(direction)
            if length == 0:
                continue
            v = direction / length * limit * rng.random() ** (1 / dimensions)
            if rng.random() <= np.exp(-.5 * np.sum((v / sigma) ** 2)):
                return v


def interpolate(points, times, query):
    q = np.asarray(query)
    i = np.clip(np.searchsorted(times, q, side='right') - 1, 0, len(times) - 2)
    a = (q - times[i]) / (times[i + 1] - times[i])
    a = np.clip(a, 0, 1)[..., None, None, None]
    # Exact endpoints must not inherit a missing neighbour.
    return np.where(a == 0, points[i], np.where(a == 1, points[i + 1], points[i] * (1-a) + points[i+1] * a))


def project(points, position, r, k, k1=0, k2=0):
    pc = (points - position) @ r.T
    z = pc[..., 2]
    xy = pc[..., :2] / np.where(z > 1e-8, z, np.nan)[..., None]
    radius = np.sum(xy * xy, axis=-1)
    xy *= (1 + k1 * radius + k2 * radius ** 2)[..., None]
    return xy * k[:2] + k[2:], z


def load_timing_profile(path):
    import hashlib
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path(__file__).resolve().parent / resolved
    raw = resolved.read_bytes()
    profile = json.loads(raw)
    if profile.get('schema_version') != 1 or profile.get('columns') != ['capture_interval_ms', 'pre_send_ms', 'transport_ms']:
        raise ValueError('Unsupported timing profile schema')
    observations = np.asarray(profile.get('observations'), float)
    if observations.ndim != 2 or observations.shape[1] != 3 or len(observations) < 2:
        raise ValueError('Timing profile requires at least two [interval,pre_send,transport] rows')
    if not np.isfinite(observations).all() or (observations[:, 0] <= 0).any() or (observations[:, 1:] < 0).any():
        raise ValueError('Timing profile contains invalid intervals or delays')
    return observations, {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                          'source': profile.get('source'), 'rows': len(observations),
                          'summary': profile.get('summary'), 'mode': 'contiguous cyclic replay, independent random offset per camera',
                          'overrides': ['camera_fps', 'capture_jitter_ms', 'processing_ms',
                                        'latency_ms', 'latency_jitter_ms', 'rolling_shutter_ms']}


def replay_camera_timing(observations, duration, rng):
    start = int(rng.integers(len(observations)))
    phase = rng.uniform(0, np.mean(observations[:, 0]) / 1000)
    events = []; indices = []; current = phase; row = start
    while current <= duration:
        events.append(current); indices.append(row)
        row = (row + 1) % len(observations)
        current += observations[row, 0] / 1000
    indices = np.asarray(indices, dtype=np.int64)
    delays = observations[indices, 1:].sum(axis=1) / 1000
    return np.asarray(events), delays, indices


FEATURES = ['u_normalized', 'v_normalized', 'ray_ox', 'ray_oy', 'ray_oz',
            'ray_dx', 'ray_dy', 'ray_dz', 'capture_minus_query_s',
            'arrival_minus_query_s', 'observed_delay_s', 'camera_id', 'hand_id', 'joint_id']


def simulate(points, times, cfg):
    cfg.validate()
    if cfg.randomize_errors:
        # Independent stream for selecting scenario severity; deterministic by seed.
        sampler = np.random.default_rng(np.random.SeedSequence([cfg.seed, 9817]))
        effective = asdict(cfg)
        ignored = {'capture_jitter_ms', 'processing_ms', 'latency_ms', 'latency_jitter_ms', 'rolling_shutter_ms'} if cfg.timing_profile else set()
        sampled = {name: float(sampler.uniform(*bounds)) for name, bounds in sorted(cfg.error_ranges.items()) if name not in ignored}
        effective.update(sampled)
        # A fixed nominal FOV and image size imply a fixed pinhole calibration.
        effective.update(randomize_errors=False, intrinsics=[], focal_std_pct=0., principal_std_px=0.)
        result = simulate(points, times, Config(**effective))
        metadata = json.loads(result['metadata'])
        metadata['randomization'] = {'enabled': True, 'distribution': 'uniform per sequence',
                                     'requested_config': asdict(cfg), 'sampled_parameters': sampled,
                                     'ignored_timing_ranges': sorted(ignored & cfg.error_ranges.keys()),
                                     'fixed_optics': ['width', 'height', 'diagonal_fov']}
        result['metadata'] = json.dumps(metadata)
        return result
    points, transform = normalize(points, cfg)
    rng = np.random.default_rng(cfg.seed)
    pose_rng = np.random.default_rng(np.random.SeedSequence([cfg.seed if cfg.pose_seed is None else cfg.pose_seed, 7283]))
    timing, timing_metadata = load_timing_profile(cfg.timing_profile) if cfg.timing_profile else (None, None)
    timing_start_rows = []
    query = np.arange(int(np.floor(times[-1] * cfg.output_fps)) + 1) / cfg.output_fps
    n = len(query)
    pixels = np.zeros((n, 3, 2, 21, 2), np.float32)
    rays = np.zeros((n, 3, 2, 21, 3), np.float32)
    valid = np.zeros((n, 3, 2, 21), bool)
    in_frame = np.zeros_like(valid)
    stamps = np.zeros((n, 3, 3))  # reported capture, arrival, physical capture
    selected = np.full((n, 3), -1, np.int64)
    origins = np.asarray(cfg.cameras, float)
    rotations, actual_positions, actual_rotations, actual_ks = [], [], [], []
    yaw_pitch_errors = []
    focal = np.hypot(cfg.width, cfg.height) / (2 * np.tan(np.deg2rad(cfg.diagonal_fov / 2)))
    ks = np.asarray(cfg.intrinsics or [[focal, focal, (cfg.width-1)/2, (cfg.height-1)/2]] * 3, dtype=float)
    for c in range(3):
        r = look_at(origins[c], cfg.targets[c])
        true_pos = origins[c] + bounded_normal_vector(pose_rng, cfg.position_std, None if cfg.position_limit_cm is None else cfg.position_limit_cm / cfg.world_unit_cm)
        true_r, yaw_pitch = camera_pose_without_roll(pose_rng, r, cfg.angle_std_deg, cfg.angle_limit_deg)
        yaw_pitch_errors.append(yaw_pitch)
        true_k = ks[c].copy()
        true_k[:2] *= np.exp(rng.normal(0, cfg.focal_std_pct / 100, 2))
        true_k[2:] += rng.normal(0, cfg.principal_std_px, 2)
        rotations.append(r); actual_positions.append(true_pos); actual_rotations.append(true_r); actual_ks.append(true_k)
        # Independent camera phase, positive ordered capture times.
        if timing is not None:
            ct, measured_delay, timing_rows = replay_camera_timing(timing, times[-1], rng)
            timing_start_rows.append(int(timing_rows[0]) if len(timing_rows) else None)
        else:
            phase = rng.uniform(0, 1/cfg.camera_fps)
            ct = np.arange(0, times[-1] + 1/cfg.camera_fps, 1/cfg.camera_fps) + phase
            ct += rng.normal(0, cfg.capture_jitter_ms / 1000, len(ct))
            ct = np.unique(ct[(ct >= 0) & (ct <= times[-1])])
        if not len(ct):
            continue
        world = interpolate(points, times, ct)
        uv, depth = project(world, true_pos, true_r, true_k, cfg.k1, cfg.k2)
        if cfg.rolling_shutter_ms and timing is None:
            for h in range(2):
                for j in range(21):
                    row = np.nan_to_num(uv[:, h, j, 1] / cfg.height, nan=0.)
                    qt = ct + np.clip(row, 0, 1) * cfg.rolling_shutter_ms / 1000
                    world[:, h, j] = interpolate(points, times, qt)[:, h, j]
                    world[qt > times[-1], h, j] = np.nan
            uv, depth = project(world, true_pos, true_r, true_k, cfg.k1, cfg.k2)
        uv += rng.normal(0, cfg.pixel_std, uv.shape)
        outliers = rng.random(uv.shape[:-1]) < cfg.outlier_prob
        uv += outliers[..., None] * rng.normal(0, cfg.outlier_std_px, uv.shape)
        ok = np.isfinite(uv).all(-1) & (depth > 0) & (uv[..., 0] >= 0) & (uv[..., 0] < cfg.width) & (uv[..., 1] >= 0) & (uv[..., 1] < cfg.height)
        frame_ok = ok.copy()
        ok &= rng.random(ok.shape) >= cfg.missing_prob
        if timing is not None:
            arrival = ct + measured_delay
        else:
            delay = np.maximum(0, rng.normal(cfg.latency_ms, cfg.latency_jitter_ms, len(ct))) / 1000
            arrival = ct + cfg.processing_ms / 1000 + delay + cfg.rolling_shutter_ms / 1000
        lost = rng.random(len(ct)) < cfg.packet_loss
        reported = ct * (1 + rng.normal(0, cfg.clock_drift_std_ppm) * 1e-6) + rng.normal(0, cfg.clock_offset_std_ms) / 1000
        # Sweep arrival order; never let an out-of-order old frame replace a newer frame.
        order = np.argsort(arrival, kind='stable'); cursor = 0; latest = -1
        for i, now in enumerate(query):
            while cursor < len(order) and arrival[order[cursor]] <= now:
                event = order[cursor]
                if not lost[event]:
                    latest = max(latest, int(event))
                cursor += 1
            if latest < 0 or now - ct[latest] > cfg.max_age_ms / 1000:
                continue
            selected[i, c] = latest
            stamps[i, c] = [reported[latest], arrival[latest], ct[latest]]
            valid[i, c] = ok[latest]
            in_frame[i, c] = frame_ok[latest]
            pixels[i, c] = np.nan_to_num(uv[latest])
            # Estimated pinhole calibration deliberately differs from the true rig.
            d = np.concatenate([(uv[latest] - ks[c, 2:]) / ks[c, :2], np.ones((2, 21, 1))], -1) @ r
            d /= np.linalg.norm(d, axis=-1, keepdims=True)
            rays[i, c] = np.nan_to_num(d)
    features = np.zeros((n, 3, 2, 21, len(FEATURES)), np.float32)
    features[..., :2] = pixels / [cfg.width, cfg.height]
    features[..., 2:5] = origins[None, :, None, None]
    features[..., 5:8] = rays
    features[..., 8] = (stamps[..., 0] - query[:, None])[:, :, None, None]
    features[..., 9] = (stamps[..., 1] - query[:, None])[:, :, None, None]
    features[..., 10] = (stamps[..., 1] - stamps[..., 0])[:, :, None, None]
    features[..., 11] = np.arange(3)[None, :, None, None]
    features[..., 12] = np.arange(2)[None, None, :, None]
    features[..., 13] = np.arange(21)[None, None, None, :]
    features[~valid] = 0
    pixels[~valid] = 0; rays[~valid] = 0
    labels = interpolate(points, times, query)
    label_mask = np.isfinite(labels).all(-1)
    starts = np.arange(0, max(0, n - cfg.window + 1), cfg.stride, dtype=np.int64)
    position_errors = np.linalg.norm(np.asarray(actual_positions) - origins, axis=1) * cfg.world_unit_cm
    relative_rotations = np.asarray(actual_rotations) @ np.transpose(rotations, (0, 2, 1))
    angle_errors = np.rad2deg(np.arccos(np.clip((np.trace(relative_rotations, axis1=1, axis2=2)-1)/2, -1, 1)))
    return dict(features=features, input_mask=valid, in_frame_mask=in_frame, pixels=pixels, ray_directions=rays,
                target_xyz=np.nan_to_num(labels).astype(np.float32), target_mask=label_mask,
                query_time=query, capture_time=stamps[..., 0], arrival_time=stamps[..., 1],
                physical_capture_time=stamps[..., 2], selected_frame=selected,
                window_starts=starts, nominal_origins=origins, nominal_rotations=np.asarray(rotations),
                nominal_intrinsics=ks, actual_origins=np.asarray(actual_positions),
                actual_rotations=np.asarray(actual_rotations), actual_intrinsics=np.asarray(actual_ks),
                sampled_yaw_pitch_errors_deg=np.asarray(yaw_pitch_errors),
                actual_roll_deg=np.rad2deg(np.arctan2(np.asarray(actual_rotations)[:,0,1], -np.asarray(actual_rotations)[:,1,1])),
                actual_position_errors_cm=position_errors, actual_angle_errors_deg=angle_errors,
                metadata=json.dumps({'schema_version': 1, 'config': asdict(cfg), 'transform': transform,
                                     'features': FEATURES, 'target': 'query-time world coordinates',
                                     'timing_replay': {**timing_metadata, 'start_rows': timing_start_rows} if timing_metadata else None,
                                     'units': {'world_unit_cm': cfg.world_unit_cm, 'xyz_to_cm': cfg.world_unit_cm,
                                               'workspace_side_cm': 2 * cfg.world_unit_cm,
                                               'pose_error_distribution': 'XYZ translation Gaussian; yaw/pitch Gaussian; optional legacy limits',
                                               'camera_rotation': 'zero roll relative to world Y up; yaw about world Y, pitch elevation',
                                               'camera_roll_enabled': False},
                                     'hand_order': 'source order, unless swap_hands enabled'}))


def save_dataset(path, result, source='demo'):
    metadata = json.loads(result['metadata']); metadata['source'] = str(source)
    with open(path, 'wb') as stream:
        np.savez_compressed(stream, **{**result, 'metadata': json.dumps(metadata)})


def training_window(data, index):
    """Flatten time/camera/hand/joint tokens; use ~mask as Transformer padding mask."""
    cfg = json.loads(str(data['metadata']))['config']
    start = int(data['window_starts'][index]); end = start + cfg['window']
    x = data['features'][start:end].copy()
    # Every observation timestamp is relative to the final prediction query.
    delta = data['query_time'][start:end] - data['query_time'][end-1]
    x[..., 8:10] += delta[:, None, None, None, None]
    mask = data['input_mask'][start:end]
    x[~mask] = 0
    return x.reshape(-1, len(FEATURES)), mask.reshape(-1), data['target_xyz'][end-1], data['target_mask'][end-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--member', help='Sequence path inside TAR archive')
    parser.add_argument('--list-sequences', action='store_true', help='List sequence candidates in --input TAR')
    parser.add_argument('--inspect', nargs='?', const='', help='Open NPZ reader, optionally with a file path')
    args = parser.parse_args()
    if args.list_sequences:
        if not args.input: parser.error('--list-sequences requires --input')
        from motion_archive import list_motion_members
        for name in list_motion_members(args.input): print(name)
        return
    if args.inspect is not None:
        from npz_reader import launch as launch_reader
        launch_reader(args.inspect or None)
        return
    cfg = Config(**json.loads(args.config.read_text(encoding='utf-8'))) if args.config else Config()
    if args.output:
        cfg.validate()
        p, t = load_motion(args.input, cfg.source_fps, args.member) if args.input else demo_motion()
        result = simulate(p, t, cfg)
        source = f'{args.input}::{args.member}' if args.member else args.input or 'synthetic demo'
        save_dataset(args.output, result, source)
        print(f'Saved {args.output}: {len(result["query_time"])} frames, {len(result["window_starts"])} windows')
    else:
        from gigahands_gui import launch
        launch(cfg, args.input, args.member)


if __name__ == '__main__':
    main()
