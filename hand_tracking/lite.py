"""HandLite: a fixed-size, export-friendly pose model for asynchronous camera events.

Split so that deployment runs three small fixed-shape networks (ONNX/ncnn) around plain
geometry (lite_runtime, numpy):
  per event, once, on arrival:
    geometry  latest ray of each camera; nominal triangulation; the event's own ray miss
    encoder   21 joint features per hand -> 2 hand tokens                  [2, dim]
  per query at time t:
    slots     each camera's slots_per_camera latest events captured within event_span_s
    calibrator  tokens pooled per camera -> camera correction                 [3, 6]
    geometry  corrected triangulation of every slot's rays; line fit to t -> anchor + flags
    decoder   42 joint queries (anchor) attend to the 2*3*K hand tokens -> offset [42, 3]
The pose is anchor + offset. Networks use only Linear/LayerNorm/GELU/MatMul/Softmax and
additive float masks. forward() computes exactly what LiteStream computes event by event.
"""
from dataclasses import dataclass, replace
import torch
from torch import nn
import torch.nn.functional as F
from .config import LiteConfig
from .constants import (NUM_CAMERAS, NUM_HANDS, NUM_JOINTS, HAND_JOINTS, FEATURE_CHANNELS, CALIBRATION_PARAMS,
                        UV, RAY_ORIGIN, RAY_DIRECTION, DELAY, TIME_UNIT_S, ARRIVAL_TOLERANCE_S)
from .contracts import ModelOutput
from .events import _gather, _miss, latest_rays, make_events, query_anchor, relative_events, sample_points
from .stream import EventStream, StreamCache

# Per joint: u,v (2), ray origin (3) and direction (3), delay (1), miss vector (3), seen (1),
# triangulated with another camera (1).
JOINT_FEATURES = 14
HAND_FEATURES = NUM_JOINTS*JOINT_FEATURES
# Anchor position (3) and query_anchor flags (5).
ANCHOR_FEATURES = 8
# Additive attention bias of an excluded key; finite so fp16 runtimes stay NaN-free.
MASK_OFF = -1e4


def key_bias(valid):
    """Boolean key validity -> additive attention bias (0 or MASK_OFF)."""
    return (valid.float()-1.)*-MASK_OFF


class Attention(nn.Module):
    """Multi-head attention with an additive bias; 4-D tensors at most (ncnn limit)."""

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.query, self.key, self.value, self.out = (nn.Linear(dim, dim) for _ in range(4))

    def forward(self, x, keys, bias=None):
        """x [N,L,d], keys [N,S,d], bias broadcastable to [N,heads,L,S] -> [N,L,d]."""
        n, l, d = x.shape
        s, h = keys.shape[1], self.heads
        q = self.query(x).reshape(n, l, h, d//h).permute(0, 2, 1, 3)
        k = self.key(keys).reshape(n, s, h, d//h).permute(0, 2, 3, 1)
        v = self.value(keys).reshape(n, s, h, d//h).permute(0, 2, 1, 3)
        scores = (q @ k)*(d//h)**-.5
        if bias is not None:
            scores = scores+bias
        weights = F.dropout(scores.softmax(-1), self.dropout, self.training)
        return self.out((weights @ v).permute(0, 2, 1, 3).reshape(n, l, d))


class Block(nn.Module):
    """Pre-norm: optional cross-attention to keys, self-attention, feed-forward."""

    def __init__(self, dim, heads, dropout, cross=True):
        super().__init__()
        self.cross = Attention(dim, heads, dropout) if cross else None
        if cross:
            self.norm_query, self.norm_keys = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.norm_self, self.norm_ff = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = Attention(dim, heads, dropout)
        self.feedforward = nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(), nn.Linear(2*dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, keys=None, bias=None):
        if self.cross is not None:
            x = x+self.drop(self.cross(self.norm_query(x), self.norm_keys(keys), bias))
        y = self.norm_self(x)
        x = x+self.drop(self.attention(y, y))
        return x+self.drop(self.feedforward(self.norm_ff(x)))


class EventEncoder(nn.Module):
    """features [N,2,21*14], camera one-hot [N,3] -> hand tokens [N,2,dim]."""

    def __init__(self, dim):
        super().__init__()
        self.joints = nn.Sequential(nn.Linear(HAND_FEATURES, dim), nn.GELU(), nn.Linear(dim, dim))
        self.hand = nn.Parameter(torch.randn(NUM_HANDS, dim)*.02)
        self.camera = nn.Linear(NUM_CAMERAS, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, features, camera):
        return self.norm(self.joints(features)+self.hand+self.camera(camera)[:, None])


class Calibrator(nn.Module):
    """tokens [N,T,dim], pool bias [N,3,T] (camera c's valid tokens 0, others MASK_OFF)
    -> correction [N,3,6] (rotation vector rad, shift world units); zero until trained."""

    def __init__(self, dim, heads, dropout, scale):
        super().__init__()
        self.query = nn.Parameter(torch.randn(NUM_CAMERAS, dim)*.02)
        self.null = nn.Parameter(torch.zeros(1, dim))
        self.norm = nn.LayerNorm(dim)
        self.pool = Attention(dim, heads, dropout)
        self.mix = Block(dim, heads, dropout, cross=False)
        self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, CALIBRATION_PARAMS))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        self.register_buffer('scale', scale)

    def forward(self, tokens, pool_bias):
        n, _, d = tokens.shape
        keys = torch.cat((self.null.expand(n, 1, d), self.norm(tokens)), 1)
        bias = torch.cat((torch.zeros_like(pool_bias[:, :, :1]), pool_bias), -1)[:, None]
        pooled = self.pool(self.query.expand(n, NUM_CAMERAS, d), keys, bias)
        return self.output(self.mix(pooled))*self.scale


class Decoder(nn.Module):
    """tokens [N,T,dim], key bias [N,T], capture-to-query gaps [N,T] (TIME_UNIT_S),
    anchor features [N,42,8] -> offset [N,42,3] from the anchor; zero until trained."""

    def __init__(self, dim, heads, blocks, dropout):
        super().__init__()
        self.joints = nn.Parameter(torch.randn(HAND_JOINTS, dim)*.02)
        self.anchor = nn.Sequential(nn.Linear(ANCHOR_FEATURES, dim), nn.GELU(), nn.Linear(dim, dim))
        self.null = nn.Parameter(torch.zeros(1, dim))
        self.time_bias = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, heads))
        self.blocks = nn.ModuleList([Block(dim, heads, dropout) for _ in range(blocks)])
        self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 3))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, tokens, token_bias, gaps, anchor_features):
        n, _, d = tokens.shape
        x = self.joints+self.anchor(anchor_features)
        keys = torch.cat((self.null.expand(n, 1, d), tokens), 1)
        bias = self.time_bias(gaps[..., None]).permute(0, 2, 1)+token_bias[:, None]           # [N,h,T]
        bias = torch.cat((torch.zeros_like(bias[..., :1]), bias), -1)[:, :, None]             # [N,h,1,1+T]
        for block in self.blocks:
            x = block(x, keys, bias)
        return self.output(x)


@dataclass
class LiteEncoded:
    """Per-event results that never change after arrival. Times on the batch origin
    (stream storage: absolute float64)."""
    origin: torch.Tensor     # [B,E,3,2,21,3] each camera's latest ray at the event's arrival
    direction: torch.Tensor  # [B,E,3,2,21,3]
    mask: torch.Tensor       # [B,E,3,2,21]
    capture: torch.Tensor    # [B,E,3] capture time of those rays
    evident: torch.Tensor    # [B,E] some own joint triangulated with another camera
    tokens: torch.Tensor     # [B,E,2,dim]

    def map(self, fn):
        return LiteEncoded(*(fn(value) for value in self.__dict__.values()))

    def append(self, other, dim=1):
        return LiteEncoded(*(torch.cat((a, b), dim) for a, b in zip(self.__dict__.values(), other.__dict__.values())))


def joint_features(raw, valid, point, ok):
    """[...,2,21,11] event features, own valid, nominal sample point/ok -> [...,2,21,14]."""
    origin, direction = raw[..., RAY_ORIGIN], raw[..., RAY_DIRECTION]
    miss = _miss(point, ok, origin, direction, valid)
    seen = (ok & valid).float()[..., None]
    x = torch.cat((raw[..., UV]*2-1, origin, direction, raw[..., DELAY]/TIME_UNIT_S, miss, valid.float()[..., None],
                   seen), -1)
    return torch.where(valid[..., None], x, 0.)


class HandLite(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or LiteConfig()
        c = self.config
        scale = torch.tensor([c.calibration_rotation_deg*torch.pi/180]*3+[c.calibration_position]*3)
        self.encoder = EventEncoder(c.dim)
        self.calibrator = Calibrator(c.dim, c.heads, c.dropout, scale) if c.calibration_head else None
        self.decoder = Decoder(c.dim, c.heads, c.blocks, c.dropout)

    @property
    def slots(self):
        return NUM_CAMERAS*self.config.slots_per_camera

    def encode_events(self, events, targets=None):
        """LiteEncoded of the target events (default: all), given every earlier event."""
        b, e = events['camera'].shape
        if targets is None:
            targets = torch.arange(e, device=events['camera'].device)
        t = targets.numel()
        with torch.autocast(device_type=events['raw'].device.type, enabled=False):
            origin, direction, mask, capture = latest_rays(events, targets, self.config.sample_max_age_s)
            point, ok, _, _ = sample_points(origin, direction, mask, capture)
            raw, valid = events['raw'][:, targets], events['valid'][:, targets]
            features = joint_features(raw, valid, point, ok)
        camera = F.one_hot(events['camera'][:, targets], NUM_CAMERAS).float()
        tokens = self.encoder(features.reshape(b*t, NUM_HANDS, HAND_FEATURES), camera.reshape(b*t, NUM_CAMERAS))
        evident = (ok & valid).flatten(2).any(-1) & events['present'][:, targets]
        return LiteEncoded(origin, direction, mask, capture, evident, tokens.reshape(b, t, NUM_HANDS, -1))

    def select(self, events, query):
        """Slots [B,Q,3K]: event indices in arrival order (padding first), and validity.
        Each camera's K latest accepted events arrived by the query, captured within span."""
        k = self.config.slots_per_camera
        b, e = events['camera'].shape
        index = torch.arange(e, device=query.device)
        usable = (events['present'][:, None] & (events['arrival'][:, None] <= query[..., None]+ARRIVAL_TOLERANCE_S)
                  & (events['capture'][:, None] >= query[..., None]-self.config.event_span_s))          # [B,Q,E]
        chosen = []
        for camera in range(NUM_CAMERAS):
            score = torch.where(usable & (events['camera'][:, None] == camera), index, -1)
            if e < k:
                score = F.pad(score, (0, k-e), value=-1)
            chosen.append(score.topk(k, dim=-1).values)
        slots = torch.cat(chosen, -1).sort(-1).values
        return slots.clamp_min(0), slots >= 0

    def decode_queries(self, events, encoded, query):
        """Pose [B,Q,2,21,3], calibration [B,Q,3,6] and whether a query had any slot [B,Q]."""
        c = self.config
        b, q = query.shape
        n, s = b*q, self.slots
        slot, valid = self.select(events, query)                                           # [B,Q,S]
        take = lambda value: _gather(value, slot).reshape(n, s, *value.shape[2:])
        camera, capture, arrival = take(events['camera']), take(events['capture']), take(events['arrival'])
        valid = valid.reshape(n, s)
        tokens = take(encoded.tokens).reshape(n, 2*s, -1)
        token_valid = valid.repeat_interleave(NUM_HANDS, -1)
        token_camera = camera.repeat_interleave(NUM_HANDS, -1)
        gaps = ((query.reshape(n, 1)-capture)/TIME_UNIT_S).clamp_min(0.).repeat_interleave(NUM_HANDS, -1)
        cameras = torch.arange(NUM_CAMERAS, device=query.device)[:, None]
        if self.calibrator is not None:
            pool = key_bias((token_camera[:, None] == cameras) & token_valid[:, None])        # [N,3,2S]
            params = self.calibrator(tokens, pool).float()
            # A camera without triangulated evidence in its slots keeps the nominal calibration.
            informative = ((camera[:, None] == cameras) & (valid & take(encoded.evident))[:, None]).any(-1)
            params = torch.where(informative[..., None], params, 0.)
        else:
            params = tokens.new_zeros(n, NUM_CAMERAS, CALIBRATION_PARAMS)
        with torch.autocast(device_type=query.device.type, enabled=False):
            point, ok, stamp, _ = sample_points(take(encoded.origin), take(encoded.direction), take(encoded.mask),
                                                take(encoded.capture), params[:, None].expand(n, s, -1, -1))
            ok = ok & valid[..., None, None]
            anchor, flags = query_anchor(point, ok, stamp, arrival, valid, query.reshape(n, 1),
                                         c.anchor_lookback_s, c.anchor_hold_s)
        features = torch.cat((anchor, flags), -1).reshape(n, HAND_JOINTS, ANCHOR_FEATURES)
        offset = self.decoder(tokens, key_bias(token_valid), gaps, features)
        pose = anchor.reshape(n, HAND_JOINTS, 3)+offset.float()
        return (pose.reshape(b, q, NUM_HANDS, NUM_JOINTS, 3), params.reshape(b, q, NUM_CAMERAS, CALIBRATION_PARAMS),
                valid.any(-1).reshape(b, q))

    def forward(self, event_features, event_valid, event_camera, event_capture, event_arrival, event_present,
                query_times, return_details=False):
        """Same inputs as HandTransformer.forward. return_details gives ModelOutput whose
        calibration is per query [B,Q,3,6] and accepted marks queries with any slot [B,Q]."""
        if event_features.ndim != 5 or event_features.shape[2:] != (NUM_HANDS, NUM_JOINTS, FEATURE_CHANNELS):
            raise ValueError('Expected event features [B,E,2,21,14]')
        if event_valid.shape != event_features.shape[:-1] or query_times.ndim != 2:
            raise ValueError('Invalid event/query dimensions')
        events = make_events(event_features, event_valid, event_camera, event_capture, event_arrival, event_present)
        pose, params, rows = self.decode_queries(events, self.encode_events(events), query_times.float())
        return ModelOutput(pose, params, rows) if return_details else pose


class LiteStream(EventStream):
    """HandLite event by event: each event is encoded once (geometry and hand tokens);
    a query runs slot selection, calibrator, corrected anchor and decoder. Equals forward()."""

    def flush(self):
        with torch.inference_mode():
            if not self.pending:
                return
            origin = self.pending[-1].arrival
            raw = self._pending_raw()
            old = self.store
            if old is not None:
                keep = old.raw['capture'] >= origin-self.keep_s
                if not keep.all():
                    old = StreamCache({key: value[keep] for key, value in old.raw.items()},
                                      old.encoded.map(lambda value: value[keep]), None)
                raw = {key: torch.cat((old.raw[key], value)) for key, value in raw.items()}
            count = len(raw['camera'])
            targets = torch.arange(count-len(self.pending), count, device=self.device)
            encoded = self.model.encode_events(relative_events(raw, origin), targets).map(lambda value: value[0])
            encoded = replace(encoded, capture=encoded.capture.double()+origin)
            if old is not None:
                encoded = old.encoded.append(encoded, dim=0)
            self.store = StreamCache(raw, encoded, None)
            self.pending.clear()

    def query(self, time):
        time = float(time)
        latest = self._latest_arrival()
        if latest is None:
            raise ValueError('No events yet')
        if time < latest:
            raise ValueError('Query before the latest arrival')
        self.flush()
        with torch.inference_mode():
            encoded = self.store.encoded.map(lambda value: value[None])
            encoded = replace(encoded, capture=(encoded.capture-time).float())
            query = torch.zeros(1, 1, device=self.device)
            return self.model.decode_queries(relative_events(self.store.raw, time), encoded, query)[0][0, 0]
