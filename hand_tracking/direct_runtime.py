"""HandDirect deployment runtime without PyTorch: slot selection around the two exported
networks, run by ncnn. Mirrors hand_tracking.direct.DirectStream.

    runtime = DirectRuntime('exports/direct_model')
    runtime.push(camera, features [2,21,14], valid [2,21], capture_time, arrival_time)
    pose = runtime.query(time)          # [2,21,3] world units

Times are absolute seconds (float64). Needs numpy and `ncnn`.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from .config import DirectConfig
from .constants import (NUM_CAMERAS, NUM_HANDS, NUM_JOINTS, MODEL_CHANNELS, UV, RAY_ORIGIN, RAY_DIRECTION, DELAY,
                        TIME_UNIT_S, STREAM_KEEP_MARGIN_S, check_event)
from .runtime import NcnnGraphs, select_slot_events

META_FILE = 'direct.json'
EXPORT_FORMAT = 2   # 2: ncnn only (pnnx), graph input shapes in the metadata
GRAPHS = ('encoder', 'query')


def direct_features(raw, valid):
    """Event features [2,21,11] (invalid joints zeroed), valid -> encoder input [2,21,10]."""
    x = np.concatenate((raw[..., UV]*2-1, raw[..., RAY_ORIGIN], raw[..., RAY_DIRECTION], raw[..., DELAY]/TIME_UNIT_S,
                        valid[..., None]), -1)
    return np.where(valid[..., None], x, 0.)


@dataclass
class _Event:
    camera: int
    capture: float
    arrival: float
    tokens: np.ndarray     # [10,dim]


class DirectRuntime:
    def __init__(self, directory, threads=1, fp16=True, graphs=None):
        """graphs: any object with run(name, *arrays), instead of loading ncnn."""
        directory = Path(directory)
        meta = json.loads((directory/META_FILE).read_text(encoding='utf-8'))
        if meta.get('architecture') != 'direct' or meta.get('export_format') != EXPORT_FORMAT:
            raise ValueError(f'{directory} is not a HandDirect export of format {EXPORT_FORMAT}')
        self.config = DirectConfig(**meta['config'])
        self.graphs = graphs if graphs is not None else NcnnGraphs(directory, GRAPHS, threads, fp16)
        self.keep_s = self.config.context_s+STREAM_KEEP_MARGIN_S
        self.reset()

    def reset(self):
        self.events = []
        self.newest_capture = [None]*NUM_CAMERAS
        self.last_arrival = None

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
        tokens = self.graphs.run('encoder', direct_features(raw, valid).reshape(NUM_HANDS, -1), np.eye(NUM_CAMERAS)[camera])
        self.events.append(_Event(camera, capture, arrival, np.asarray(tokens)))
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
        slots = select_slot_events(self.events, time, self.config.slots_per_camera, self.config.event_span_s)
        per, dim = self.config.event_tokens, self.config.dim
        present = np.array([event is not None for event in slots])
        tokens = np.stack([event.tokens if event is not None else np.zeros((per, dim)) for event in slots])
        ages = np.array([max(time-event.capture, 0.)/TIME_UNIT_S if event is not None else 0. for event in slots])
        joints = self.graphs.run('query', tokens.reshape(-1, dim), np.repeat(present, per), np.repeat(ages, per))
        return np.asarray(joints).reshape(NUM_HANDS, NUM_JOINTS, 3)
