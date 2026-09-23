"""Arrival-ordered event buffering and inference caches, separate from model math."""
from dataclasses import dataclass, replace
from typing import NamedTuple
import torch
from .contracts import RawEvents, EncodedEvents, Evidence
from .events import relative_events

STREAM_KEEP_MARGIN_S = .5


class PendingEvent(NamedTuple):
    features: torch.Tensor
    valid: torch.Tensor
    camera: int
    capture: float
    arrival: float


@dataclass
class StreamCache:
    """Unbatched tensors with absolute float64 capture, arrival and sample times."""
    raw: RawEvents
    encoded: EncodedEvents
    evidence: Evidence | None

    def select(self, keep):
        return StreamCache({key: value[keep] for key, value in self.raw.items()},
                           self.encoded.map(lambda value: value[keep]),
                           None if self.evidence is None else self.evidence.map(lambda value: value[keep]))


class EventStream:
    """Cache each accepted camera event once; query arbitrary current/future times.

    Events must arrive in order. Empty detections replace older camera rays;
    duplicate or older captures are dropped, matching the batch event policy.
    push() owns its input tensors. query()/flush() batch all pending encodings.
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
        camera = int(camera)
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
        previous = None if old is None or old.evidence is None else old.evidence.map(lambda value: value[None])
        encoded, _, evidence = self.model.encode_events(relative_events(raw, origin), targets, previous)
        encoded = encoded.map(lambda value: value[0])
        encoded = replace(encoded, stamp=encoded.stamp.double()+origin)
        evidence = None if evidence is None else evidence.map(lambda value: value[0])
        if old is not None:
            encoded = old.encoded.append(encoded, dim=0)
            if evidence is not None:
                evidence = old.evidence.append(evidence, dim=0)
        self.store = StreamCache(raw, encoded, evidence)
        self.pending.clear()

    @torch.inference_mode()
    def query(self, time):
        time = float(time)
        latest = self._latest_arrival()
        if latest is None:
            raise ValueError('No events yet')
        if time < latest:
            raise ValueError('Query before the latest arrival')
        self.flush()
        stored = self.store
        encoded = stored.encoded.map(lambda value: value[None])
        encoded = replace(encoded, stamp=(encoded.stamp-time).float())
        query = torch.zeros(1, 1, device=self.device)
        return self.model.decode_queries(relative_events(stored.raw, time), encoded, query)[0,0]
