"""Camera events: acceptance, packing and slot selection. No learned parameters."""
import torch
import torch.nn.functional as F
from .constants import NUM_CAMERAS, MODEL_CHANNELS, ARRIVAL_TOLERANCE_S
from .contracts import EventBatch, RawEvents


def accepted_captures(camera, capture, present):
    """Keep strictly newer captures per camera in arrival order; padding is ignored.

    This is the batch equivalent of EventStream's per-camera high-water mark: an event is
    dropped when an earlier present event of its camera has a capture at least as new.
    Pairwise [B,E,E] instead of a running maximum (cummax has no ONNX operator).
    Comparing the original dtype preserves float64 stream timestamp precision.
    """
    present = present.bool() & (camera >= 0) & (camera < NUM_CAMERAS) & torch.isfinite(capture)
    index = torch.arange(camera.shape[1], device=camera.device)
    earlier = index[None, :, None] > index[None, None, :]                                # [1,E(i),E(j)]: j before i
    superseded = (earlier & present[:, None, :] & (camera[:, :, None] == camera[:, None, :])
                  & (capture[:, None, :] >= capture[:, :, None])).any(-1)
    return present & ~superseded


def make_events(features, valid, camera, capture, arrival, present) -> EventBatch:
    present = accepted_captures(camera, capture, present)
    return _pack_events(features, valid, camera, capture, arrival, present)


def _pack_events(features, valid, camera, capture, arrival, present) -> EventBatch:
    """Canonical tensor layout after the caller has resolved event acceptance."""
    valid = valid.bool() & present[..., None, None]
    raw = torch.where(valid[..., None], features[..., :MODEL_CHANNELS].float(), 0.)
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
