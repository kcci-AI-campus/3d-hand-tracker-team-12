"""Arrival-ordered event buffering for streaming inference, separate from model math.

EventStream owns pushed frames, drops stale captures and prunes old events; each model's
stream subclass says how its per-event encoding moves between relative and absolute time
(to_absolute) and how a query is decoded from the stored events (decode)."""
from dataclasses import dataclass
from typing import Any, NamedTuple
import torch
from .contracts import RawEvents
from .constants import STREAM_KEEP_MARGIN_S, check_event
from .events import relative_events


class PendingEvent(NamedTuple):
    features: torch.Tensor
    valid: torch.Tensor
    camera: int
    capture: float
    arrival: float


@dataclass
class StreamCache:
    """Stored events (absolute float64 capture and arrival times) and their encodings:
    any object with map(fn) and append(other, dim)."""
    raw: RawEvents
    encoded: Any

    def select(self, keep):
        return StreamCache({key: value[keep] for key, value in self.raw.items()},
                           self.encoded.map(lambda value: value[keep]))


class EventStream:
    """Encode each accepted camera event once; query any time from the latest arrival on.

    Events must arrive in order. Empty detections replace older camera rays; duplicate or
    older captures are dropped, matching the batch event policy (events.accepted_captures).
    push() owns its input tensors. query()/flush() encode all pending events in one batch.
    """

    def __init__(self, model):
        self.model = model.eval()
        self.config = model.config
        self.device = next(model.parameters()).device
        # Pruning keeps a margin beyond the exact context a new event can reach.
        self.keep_s = self.config.context_s + STREAM_KEEP_MARGIN_S
        self.reset()

    def reset(self):
        self.store: StreamCache | None = None
        self.pending: list[PendingEvent] = []
        self.newest_capture: dict[int, float] = {}

    def __len__(self):
        stored = 0 if self.store is None else len(self.store.raw['camera'])
        return stored + len(self.pending)

    def _latest_arrival(self):
        if self.pending:
            return self.pending[-1].arrival
        return None if self.store is None else float(self.store.raw['arrival'][-1])

    def push(self, camera, features, valid, capture_time, arrival_time):
        """Own a frame; return False when its capture does not advance that camera."""
        arrival_time = float(arrival_time)
        capture_time = float(capture_time)
        camera = check_event(camera, torch.as_tensor(features).shape, torch.as_tensor(valid).shape)
        latest = self._latest_arrival()
        if latest is not None and arrival_time < latest:
            raise ValueError('Events must be pushed in arrival order')
        if capture_time <= self.newest_capture.get(camera, float('-inf')):
            return False
        event = PendingEvent(torch.as_tensor(features, dtype=torch.float32, device=self.device).clone(),
                             torch.as_tensor(valid, dtype=torch.bool, device=self.device).clone(),
                             camera, capture_time, arrival_time)
        self.newest_capture[camera] = capture_time
        self.pending.append(event)
        return True

    def _pending_raw(self):
        return RawEvents(
            features=torch.stack([event.features for event in self.pending]),
            valid=torch.stack([event.valid for event in self.pending]),
            camera=torch.tensor([event.camera for event in self.pending], device=self.device),
            capture=torch.tensor([event.capture for event in self.pending], dtype=torch.float64, device=self.device),
            arrival=torch.tensor([event.arrival for event in self.pending], dtype=torch.float64, device=self.device))

    @torch.inference_mode()
    def flush(self):
        """Encode pending events transactionally: a failure leaves the buffer intact."""
        if not self.pending:
            return
        origin = self.pending[-1].arrival
        raw = self._pending_raw()
        old = self.store
        if old is not None:
            keep = old.raw['capture'] >= origin-self.keep_s
            if not keep.all():
                old = old.select(keep)
            raw = {key: torch.cat((old.raw[key], value)) for key, value in raw.items()}
        count = len(raw['camera'])
        targets = torch.arange(count-len(self.pending), count, device=self.device)
        encoded = self.model.encode_events(relative_events(raw, origin), targets).map(lambda value: value[0])
        encoded = self.to_absolute(encoded, origin)
        if old is not None:
            encoded = old.encoded.append(encoded, dim=0)
        self.store = StreamCache(raw, encoded)
        self.pending.clear()

    @torch.inference_mode()
    def query(self, time):
        """Pose [2,21,3] at time (not before the latest arrival)."""
        time = float(time)
        latest = self._latest_arrival()
        if latest is None:
            raise ValueError('No events yet')
        if time < latest:
            raise ValueError('Query before the latest arrival')
        self.flush()
        return self.decode(time)

    def to_absolute(self, encoded, origin):
        """A new unbatched encoding with its times on origin -> absolute float64 times."""
        return encoded

    def decode(self, time):
        raise NotImplementedError
