"""Event windows for the time-based model, read from simulator/GigaHands NPZ exports.
Clip-local streaming avoids repeatedly decompressing the same NPZ per window. Samples
follow contracts.WindowSample."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from .config import LiteConfig, SamplingConfig
from .constants import NUM_CAMERAS, NUM_HANDS, NUM_JOINTS, FEATURE_CHANNELS, CALIBRATION_PARAMS, RAY_ORIGIN, \
    RAY_DIRECTION
from .contracts import WindowSample

SPLITS = ('train', 'val', 'test')
# NPZ arrays a clip keeps: model inputs, targets and window indexing.
CLIP_ARRAYS = ('features', 'input_mask', 'target_xyz', 'target_mask', 'query_time', 'window_starts')
# Frame identity and model-visible (reported) timing, used only to recover events.
TIMING_ARRAYS = ('selected_frame', 'capture_time', 'arrival_time')
CALIBRATION_ARRAYS = ('actual_rotations', 'nominal_rotations', 'actual_origins', 'nominal_origins')
# Target frames: 'rig' removes each clip's rig-wide error, which no camera observation
# reveals; 'world' keeps the simulator's true world coordinates.
TARGET_FRAMES = ('rig', 'world')
# Prior scale of the per-camera corrections the rig frame keeps small (the calibrator's).
RIG_FRAME_ROTATION_STD_DEG = 5.
RIG_FRAME_POSITION_STD = .1
# read_clip's arrays that make_sample uses; the cache (cached_clip) keeps only these.
SAMPLE_ARRAYS = ('event_features', 'event_valid', 'event_camera', 'event_capture', 'event_arrival', 'target_xyz',
                 'target_mask', 'query_time', 'window_starts', 'calibration_target', 'rig_target',
                 'rig_calibration_target')
CACHE_VERSION = 1   # bump when read_clip's output changes


def read_manifest(root):
    """manifest.jsonl rows, checked for valid paths and no participant/source train-val leakage."""
    root = Path(root).resolve()
    lines = (root/'manifest.jsonl').read_text(encoding='utf-8').splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    people = {split: set() for split in SPLITS}
    sources = {split: set() for split in SPLITS}
    files = set()
    for row in rows:
        split = row.get('split')
        if split not in people:
            raise ValueError('Manifest requires train/val/test records without errors')
        path = (root/row['file']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Invalid dataset path')
        if row['file'] in files:
            raise ValueError('Duplicate output path')
        files.add(row['file'])
        people[split].add(row['participant'])
        sources[split].add(row['source'])
    if not people['train'] or not people['val']:
        raise ValueError('Both train and val must be present')
    for i, first in enumerate(SPLITS):
        for second in SPLITS[i+1:]:
            if people[first] & people[second] or sources[first] & sources[second]:
                raise ValueError('Participant or source leakage between splits')
    return rows


def rotation_log(matrix):
    """Rotation matrix -> rotation vector (angles well below 180 degrees)."""
    angle = np.arccos(np.clip((np.trace(matrix)-1)/2, -1, 1))
    axis = np.array([matrix[2,1]-matrix[1,2], matrix[0,2]-matrix[2,0], matrix[1,0]-matrix[0,1]])
    return axis*(angle/(2*np.sin(angle)) if angle > 1e-8 else .5)


def rotation_exp(vector):
    """Rotation vector [3] -> matrix (Rodrigues)."""
    angle = np.linalg.norm(vector)
    if angle < 1e-12:
        return np.eye(3)
    k = np.cross(np.eye(3), vector/angle)
    return np.eye(3)+np.sin(angle)*k+(1-np.cos(angle))*k@k


def rig_frame(data, steps=10, tolerance=1e-10):
    """Similarity (rotation [3,3], shift [3], scale) from the true world to the clip's rig
    frame, or None without the simulator rig (real captures).

    Rays only reveal the cameras relative to each other: rotating, moving or scaling the
    whole rig with the hands changes no observation. The rig frame is the true world moved
    by the similarity that leaves the smallest per-camera corrections from the nominal rig
    (rotation / RIG_FRAME_ROTATION_STD_DEG and shift / RIG_FRAME_POSITION_STD, least
    squares, Gauss-Newton): the nominal rig as it was actually placed. Hands keep their
    position around the nominal rig's origin; only the unobservable rig-wide error goes.
    """
    if not all(key in data for key in CALIBRATION_ARRAYS):
        return None
    actual, nominal, actual_origin, nominal_origin = (np.asarray(data[key], np.float64) for key in CALIBRATION_ARRAYS)
    corrections = [a.T@n for a, n in zip(actual, nominal)]                  # calibration_target's rotations
    scales = np.r_[[np.deg2rad(RIG_FRAME_ROTATION_STD_DEG)]*3, [RIG_FRAME_POSITION_STD]*3]

    def residuals(x):
        rotation, shift, scale = rotation_exp(x[:3]), x[3:6], np.exp(x[6])
        rows = [np.r_[rotation_log(rotation@c), scale*rotation@o+shift-n]/scales
                for c, o, n in zip(corrections, actual_origin, nominal_origin)]
        return np.concatenate(rows)

    x = np.zeros(7)
    for _ in range(steps):
        r = residuals(x)
        jacobian = np.stack([(residuals(x+e)-residuals(x-e))/2e-6 for e in np.eye(7)*1e-6], -1)
        step = np.linalg.lstsq(jacobian, r, rcond=None)[0]
        x = x-step
        if np.abs(step).max() < tolerance:
            break
    return rotation_exp(x[:3]), x[3:6], float(np.exp(x[6]))


def calibration_target(data):
    """Simulator's true correction of the nominal rays [3,6], NaN when unknown (real captures).

    Same convention as geometry.apply_calibration: world-axis rotation about the camera
    centre (R_actual^T R_nominal, since rays are pixel @ R) and centre shift.
    Supervision only; never a model input.
    """
    if not all(key in data for key in CALIBRATION_ARRAYS):
        return np.full((NUM_CAMERAS, CALIBRATION_PARAMS), np.nan, np.float32)
    rows = [np.concatenate([rotation_log(actual.T@nominal), actual_origin-nominal_origin])
            for actual, nominal, actual_origin, nominal_origin in zip(*(data[key] for key in CALIBRATION_ARRAYS))]
    return np.asarray(rows, np.float32)


def to_rig_frame(target, calibration, data, frame):
    """target [...,3] and calibration target [3,6] in the rig frame (see rig_frame): the hands
    moved by the similarity, the corrections those that turn the nominal rays into the moved
    true rays. Unchanged for real captures, which have no simulator rig."""
    if frame is None:
        return target, calibration
    rotation, shift, scale = frame
    nominal_origin = np.asarray(data['nominal_origins'], np.float64)
    actual_origin = np.asarray(data['actual_origins'], np.float64)
    moved = np.asarray([np.r_[rotation_log(rotation@rotation_exp(row[:3])), scale*rotation@origin+shift-nominal]
                        for row, origin, nominal in zip(calibration.astype(np.float64), actual_origin, nominal_origin)])
    return (scale*target@rotation.T+shift).astype(np.float32), moved.astype(np.float32)


def read_clip(path):
    with np.load(path, allow_pickle=False) as data:
        clip = {key: data[key].copy() for key in CLIP_ARRAYS}
        clip['calibration_target'] = calibration_target(data)
        frame = rig_frame(data)
        clip['rig_target'], clip['rig_calibration_target'] = to_rig_frame(clip['target_xyz'], clip['calibration_target'],
                                                                           data, frame)
        timing = {key: data[key].copy() for key in TIMING_ARRAYS}
        meta = json.loads(str(data['metadata']))
    clip['window'] = int(meta['config']['window'])
    clip['world_unit_cm'] = float(meta['units']['world_unit_cm'])
    clip.update(clip_events(clip, timing))
    return clip


def cached_clip(root, file, cache=None):
    """make_sample's arrays of read_clip(root/file). With a cache folder, the first read also
    writes them uncompressed to cache/file and later reads load that instead, skipping the
    NPZ decompression and the rig-frame fit; an entry is rebuilt when its source changes."""
    path = Path(root)/file
    if cache:
        entry = Path(cache)/file
        source = path.stat()
        stamp = np.array([CACHE_VERSION, source.st_size, source.st_mtime_ns])
        try:
            with np.load(entry, allow_pickle=False) as data:
                if np.array_equal(data['stamp'], stamp):
                    clip = {key: data[key] for key in data.files if key != 'stamp'}
                    clip['window'], clip['world_unit_cm'] = int(clip['window']), float(clip['world_unit_cm'])
                    return clip
        except (OSError, KeyError, ValueError):   # missing, stale format or a partial write
            pass
    clip = read_clip(path)
    clip = {key: clip[key] for key in SAMPLE_ARRAYS + ('window', 'world_unit_cm') if key in clip}
    if cache:
        entry.parent.mkdir(parents=True, exist_ok=True)
        partial = entry.with_name(f'{entry.name}.{os.getpid()}.tmp')
        with partial.open('wb') as f:
            np.savez(f, stamp=stamp, **clip)
        os.replace(partial, entry)   # readers never see a half-written entry
    return clip


def clip_events(clip, timing):
    """Camera frames (events) recovered from the query-grid export, sorted by arrival.

    Each query holds every camera's selected (latest captured, arrived) frame; a frame is
    an event at the first query that selects it, including frames in which no joint was
    detected (they still replace that camera's older rays, as live). Stale out-of-order
    arrivals never get selected, as a stream drops them. A frame superseded before any
    query (two arrivals of one camera between queries) is not recoverable from this format.
    """
    selected = timing['selected_frame']
    new = selected >= 0
    new[1:] &= selected[1:] != selected[:-1]
    frame, camera = np.nonzero(new)
    arrival = timing['arrival_time'][frame, camera].astype(np.float64)
    capture = timing['capture_time'][frame, camera].astype(np.float64)
    # Ties keep (query, camera) order; the model treats input order as arrival order.
    order = np.argsort(arrival, kind='stable')
    return dict(event_features=clip['features'][frame, camera][order],
                event_valid=clip['input_mask'][frame, camera][order],
                event_camera=camera[order].astype(np.int64), event_capture=capture[order],
                event_arrival=arrival[order])


def make_sample(clip, window_index, context_s=None, max_events=SamplingConfig.max_events,
                target_frame=SamplingConfig.target_frame, queries=0, seen_span_s=None) -> WindowSample:
    """Time-based window: the events arrived by the window's last query and captured at most
    context_s before its first query, and the window's query times with their targets.

    Times are seconds relative to the last query. Events are padded to max_events
    (event_present False); if more arrive, the oldest are dropped. target is in target_frame ('rig' or 'world', see
    rig_frame; a clip without the simulator rig has no rig-wide error to remove);
    target_world always in the world frame. queries > 0 keeps only the window's last
    queries (the evaluated last query included) and the events their history needs.
    With seen_span_s (the model's event_span_s), a hand no event captured within it before a
    query (and arrived by it) detected has no target there: nothing in the input shows it.
    """
    if target_frame not in TARGET_FRAMES:
        raise ValueError(f'target_frame must be one of {TARGET_FRAMES}')
    if context_s is None:
        context_s = LiteConfig().context_s
    end = int(clip['window_starts'][window_index])+clip['window']
    if queries < 0 or queries > clip['window']:
        raise ValueError(f"queries must be in [0, {clip['window']}]")
    start = end-queries if queries else end-clip['window']
    query = clip['query_time'][start:end].astype(np.float64)
    last = query[-1]
    arrival, capture = clip['event_arrival'], clip['event_capture']
    chosen = np.nonzero((arrival <= last+1e-9) & (capture >= query[0]-context_s))[0][-max_events:]
    n = len(chosen)
    features = np.zeros((max_events, NUM_HANDS, NUM_JOINTS, FEATURE_CHANNELS), np.float32)
    valid = np.zeros((max_events, NUM_HANDS, NUM_JOINTS), bool)
    camera = np.zeros(max_events, np.int64)
    present = np.zeros(max_events, bool)
    event_capture = np.zeros(max_events)
    event_arrival = np.zeros(max_events)
    features[:n] = clip['event_features'][chosen]
    valid[:n] = clip['event_valid'][chosen]
    camera[:n] = clip['event_camera'][chosen]
    present[:n] = True
    event_capture[:n] = capture[chosen]-last
    event_arrival[:n] = arrival[chosen]-last
    target_mask = clip['target_mask'][start:end].copy()
    if seen_span_s is not None:
        times = (query-last)[:, None]
        usable = (event_arrival[None, :n] <= times+1e-9) & (event_capture[None, :n] >= times-seen_span_s)
        seen = (usable[..., None] & valid[None, :n].any(-1)).any(1)          # [queries, hands]
        target_mask &= seen[..., None]
    world = clip['target_xyz'][start:end].copy()
    if target_frame == 'rig' and 'rig_target' in clip:
        target, calibration = clip['rig_target'][start:end].copy(), clip['rig_calibration_target'].copy()
    else:
        target, calibration = world.copy(), clip['calibration_target'].copy()
    features[~valid] = 0
    if np.any(event_arrival[present] > 1e-6):
        raise ValueError('Future arrival in historical input')
    as_float = lambda value: torch.from_numpy(value.astype(np.float32))
    return WindowSample(event_features=torch.from_numpy(features), event_valid=torch.from_numpy(valid),
                        event_camera=torch.from_numpy(camera), event_present=torch.from_numpy(present),
                        event_capture=as_float(event_capture), event_arrival=as_float(event_arrival),
                        query_times=as_float(query-last), target=as_float(target), target_world=as_float(world),
                        target_mask=torch.from_numpy(target_mask),
                        calibration_target=as_float(calibration),
                        world_unit_cm=torch.tensor(clip['world_unit_cm'], dtype=torch.float32))


def trim_padding(batch):
    """Batch with the event axis cut to its largest present count. make_sample puts padding
    after the events and the model ignores it, so outputs are unchanged; the event axis is
    then as long as the batch needs (max_events is a cap, not a cost)."""
    present = batch['event_present']
    if present.ndim != 2:
        return batch
    used = max(int(present.sum(1).max()), 1)
    if used == present.shape[1] or present[:, used:].any():
        return batch
    return {key: value[:, :used] if key.startswith('event_') else value for key, value in batch.items()}


class HandWindows(IterableDataset):
    """Windows of one split. Train: shuffled clips and random windows, reseeded per epoch
    (set .epoch). Val: fixed, evenly spaced windows. No augmentation: the camera rig is
    fixed, so its world coordinates are worth learning."""

    def __init__(self, root, split, windows_per_clip=16, seed=42, max_clips=0, context_s=None, max_events=SamplingConfig.max_events, target_frame=SamplingConfig.target_frame,
                 queries=0, cache=None, seen_span_s=None):
        super().__init__()
        if context_s is None:
            context_s = LiteConfig().context_s
        if (split not in SPLITS or windows_per_clip < 0 or max_clips < 0 or context_s < 0 or max_events < 1
                or target_frame not in TARGET_FRAMES or queries < 0):
            raise ValueError('Invalid sampling settings')
        self.root = Path(root)
        self.rows = [row for row in read_manifest(root) if row['split'] == split and row['windows'] > 0]
        if max_clips:
            self.rows = self.rows[:max_clips]
        if not self.rows:
            raise ValueError('No usable clips')
        self.split, self.windows_per_clip, self.seed, self.epoch = split, windows_per_clip, seed, 0
        self.context_s, self.max_events, self.target_frame, self.queries = context_s, max_events, target_frame, queries
        self.cache = cache   # folder for cached_clip; None reads the NPZ every time
        self.seen_span_s = seen_span_s   # see make_sample; None keeps every labelled hand

    def _windows(self, count):
        return min(count, self.windows_per_clip) if self.windows_per_clip else count

    def __len__(self):
        return sum(self._windows(row['windows']) for row in self.rows)

    def __iter__(self):
        training = self.split == 'train'
        indices = np.arange(len(self.rows))
        if training:
            np.random.default_rng(self.seed+self.epoch).shuffle(indices)
        worker = get_worker_info()
        if worker:
            indices = indices[worker.id::worker.num_workers]
        for index in indices:
            row = self.rows[index]
            clip = cached_clip(self.root, row['file'], self.cache)
            count = len(clip['window_starts'])
            if count != row['windows']:
                raise ValueError('Manifest window count differs from NPZ')
            if training:
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index)]))
                chosen = rng.choice(count, self._windows(count), replace=False)
            else:
                chosen = np.linspace(0, count-1, self._windows(count), dtype=int)
            for window in chosen:
                sample = make_sample(clip, window, self.context_s, self.max_events, self.target_frame,
                                     self.queries, self.seen_span_s)
                # Event-less windows cannot constrain an observation-based estimator;
                # metrics are reported on the final query, so it needs a target.
                if not sample['event_present'].any() or not sample['target_mask'][-1].any():
                    continue
                yield sample
