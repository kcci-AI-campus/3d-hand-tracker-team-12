"""Event windows for the time-based model, read from simulator/GigaHands NPZ exports.
Clip-local streaming avoids repeatedly decompressing the same NPZ per window. Samples
follow contracts.WindowSample."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from .config import ModelConfig, SamplingConfig
from .constants import NUM_CAMERAS, NUM_HANDS, NUM_JOINTS, FEATURE_CHANNELS, CALIBRATION_PARAMS, RAY_ORIGIN, \
    RAY_DIRECTION
from .contracts import WindowSample

SPLITS = ('train', 'val')
# NPZ arrays a clip keeps: model inputs, targets and window indexing.
CLIP_ARRAYS = ('features', 'input_mask', 'target_xyz', 'target_mask', 'query_time', 'window_starts')
# Frame identity and model-visible (reported) timing, used only to recover events.
TIMING_ARRAYS = ('selected_frame', 'capture_time', 'arrival_time')
CALIBRATION_ARRAYS = ('actual_rotations', 'nominal_rotations', 'actual_origins', 'nominal_origins')


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
            raise ValueError('Manifest requires train/val records without errors')
        path = (root/row['file']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Invalid dataset path')
        if row['file'] in files:
            raise ValueError('Duplicate output path')
        files.add(row['file'])
        people[split].add(row['participant'])
        sources[split].add(row['source'])
    if not all(people.values()):
        raise ValueError('Both train and val must be present')
    if people['train'] & people['val'] or sources['train'] & sources['val']:
        raise ValueError('Participant or source leakage between train and val')
    return rows


def rotation_log(matrix):
    """Rotation matrix -> rotation vector (angles well below 180 degrees)."""
    angle = np.arccos(np.clip((np.trace(matrix)-1)/2, -1, 1))
    axis = np.array([matrix[2,1]-matrix[1,2], matrix[0,2]-matrix[2,0], matrix[1,0]-matrix[0,1]])
    return axis*(angle/(2*np.sin(angle)) if angle > 1e-8 else .5)


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


def read_clip(path):
    with np.load(path, allow_pickle=False) as data:
        clip = {key: data[key].copy() for key in CLIP_ARRAYS}
        clip['calibration_target'] = calibration_target(data)
        timing = {key: data[key].copy() for key in TIMING_ARRAYS}
        meta = json.loads(str(data['metadata']))
    clip['window'] = int(meta['config']['window'])
    clip['world_unit_cm'] = float(meta['units']['world_unit_cm'])
    clip.update(clip_events(clip, timing))
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


def random_rigid(rng, max_degrees, max_shift):
    """Random rotation (axis uniform, angle up to max_degrees) and translation per axis."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(rng.uniform(-max_degrees, max_degrees))
    k = np.array([[0,-axis[2],axis[1]], [axis[2],0,-axis[0]], [-axis[1],axis[0],0]])
    rotation = np.eye(3)+np.sin(angle)*k+(1-np.cos(angle))*k@k
    return rotation, rng.uniform(-max_shift, max_shift, 3)


def _apply_rigid(features, target, calibration, rigid):
    """Move rig and hands together; pixels are unchanged by a shared rigid transform."""
    rotation, shift = rigid
    features[...,RAY_ORIGIN] = features[...,RAY_ORIGIN]@rotation.T+shift
    features[...,RAY_DIRECTION] = features[...,RAY_DIRECTION]@rotation.T
    # World-axis corrections rotate with the rig; the shift of a difference cancels.
    calibration[:,:3] = calibration[:,:3]@rotation.T
    calibration[:,3:] = calibration[:,3:]@rotation.T
    return target@rotation.T+shift


def make_sample(clip, window_index, rigid=None, context_s=None, max_events=SamplingConfig.max_events) -> WindowSample:
    """Time-based window: the events arrived by the window's last query and captured at most
    context_s before its first query, and the window's query times with their targets.

    Times are seconds relative to the last query. Events are padded to max_events
    (event_present False); if more arrive, the oldest are dropped. rigid=(R, shift)
    moves rig and hands together.
    """
    if context_s is None:
        context_s = ModelConfig().context_s
    start = int(clip['window_starts'][window_index])
    end = start+clip['window']
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
    target = clip['target_xyz'][start:end].copy()
    calibration = clip['calibration_target'].copy()
    if rigid is not None:
        target = _apply_rigid(features, target, calibration, rigid)
    features[~valid] = 0
    if np.any(event_arrival[present] > 1e-6):
        raise ValueError('Future arrival in historical input')
    as_float = lambda value: torch.from_numpy(value.astype(np.float32))
    return WindowSample(event_features=torch.from_numpy(features), event_valid=torch.from_numpy(valid),
                        event_camera=torch.from_numpy(camera), event_present=torch.from_numpy(present),
                        event_capture=as_float(event_capture), event_arrival=as_float(event_arrival),
                        query_times=as_float(query-last), target=as_float(target),
                        target_mask=torch.from_numpy(clip['target_mask'][start:end].copy()),
                        calibration_target=as_float(calibration),
                        world_unit_cm=torch.tensor(clip['world_unit_cm'], dtype=torch.float32))


class HandWindows(IterableDataset):
    """Windows of one split. Train: shuffled clips, random windows and optional rig
    augmentation, reseeded per epoch (set .epoch). Val: fixed, evenly spaced windows."""

    def __init__(self, root, split, windows_per_clip=16, seed=42, max_clips=0, rig_rotate_deg=0., rig_shift=0.,
                 context_s=None, max_events=SamplingConfig.max_events):
        super().__init__()
        if context_s is None:
            context_s = ModelConfig().context_s
        if (split not in SPLITS or windows_per_clip < 0 or max_clips < 0 or context_s < 0 or max_events < 1
                or min(rig_rotate_deg, rig_shift) < 0):
            raise ValueError('Invalid sampling settings')
        if split == 'val' and (rig_rotate_deg or rig_shift):
            raise ValueError('Rig augmentation is train-only')
        self.root = Path(root)
        self.rows = [row for row in read_manifest(root) if row['split'] == split and row['windows'] > 0]
        if max_clips:
            self.rows = self.rows[:max_clips]
        if not self.rows:
            raise ValueError('No usable clips')
        self.split, self.windows_per_clip, self.seed, self.epoch = split, windows_per_clip, seed, 0
        self.rig_rotate_deg, self.rig_shift = rig_rotate_deg, rig_shift
        self.context_s, self.max_events = context_s, max_events

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
        augment = training and (self.rig_rotate_deg or self.rig_shift)
        for index in indices:
            row = self.rows[index]
            clip = read_clip(self.root/row['file'])
            count = len(clip['window_starts'])
            if count != row['windows']:
                raise ValueError('Manifest window count differs from NPZ')
            if training:
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index)]))
                chosen = rng.choice(count, self._windows(count), replace=False)
            else:
                chosen = np.linspace(0, count-1, self._windows(count), dtype=int)
            for window in chosen:
                rigid = random_rigid(rng, self.rig_rotate_deg, self.rig_shift) if augment else None
                sample = make_sample(clip, window, rigid, self.context_s, self.max_events)
                # Event-less windows cannot constrain an observation-based estimator;
                # metrics are reported on the final query, so it needs a target.
                if not sample['event_present'].any() or not sample['target_mask'][-1].any():
                    continue
                yield sample
