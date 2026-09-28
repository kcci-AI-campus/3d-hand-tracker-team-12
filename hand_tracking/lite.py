"""HandLite: a fixed-size, export-friendly pose model for asynchronous camera events.

Split so that deployment runs three small fixed-shape networks (ONNX/ncnn) around plain
geometry (lite_runtime, numpy):
  per event, once, on arrival:
    geometry  latest ray of each camera; nominal triangulation; the event's own ray miss
    encoder   21 joint features per hand -> per hand one token per finger (wrist and its
              4 joints), or one token without finger_tokens            [event_tokens, dim]
  per query at time t:
    slots     each camera's slots_per_camera latest events captured within event_span_s
    calibrator  tokens pooled per camera -> camera correction                 [3, 6]
    geometry  corrected triangulation of every slot's rays (one outlier ray dropped); line fit
              to t -> anchor + flags;
              a joint seen by one camera: on its newest ray at a remembered depth
    decoder   42 joint queries (anchor) attend to the event_tokens*3*K tokens, each tagged with its
              capture-to-query gap -> offset [42, 3]
The pose is anchor + offset. Networks use only Linear/LayerNorm/GELU/MatMul/Softmax and
additive float masks. forward() computes exactly what LiteStream computes event by event.
"""
from dataclasses import dataclass, replace
import torch
from torch import nn
import torch.nn.functional as F
from .config import LiteConfig
from .constants import (NUM_CAMERAS, NUM_HANDS, NUM_JOINTS, HAND_JOINTS, FINGERS, FINGER_JOINTS, FEATURE_CHANNELS,
                        CALIBRATION_PARAMS, UV, RAY_ORIGIN, RAY_DIRECTION, DELAY, TIME_UNIT_S, ARRIVAL_TOLERANCE_S, MASK_OFF)
from .contracts import ModelOutput
from .events import (_gather, _gather_joint, _miss, latest_rays, make_events, query_anchor, relative_events,
                     select_slots,
                     sample_points)
from .layers import Attention, Block, EventEncoder, key_bias
from .stream import EventStream

# Per joint: u,v (2), ray origin (3) and direction (3), delay (1), miss vector (3), seen (1),
# triangulated with another camera (1).
JOINT_FEATURES = 14
HAND_FEATURES = NUM_JOINTS*JOINT_FEATURES
# Margin on anchor_hold_s when dropping slots too old for the anchor (float32 relative times).
REACH_MARGIN_S = 1e-3


class Calibrator(nn.Module):
    """tokens [N,T,dim], pool bias [N,3,T] (camera c's valid tokens 0, others MASK_OFF)
    -> correction [N,3,6] (rotation vector rad, shift world units); zero until trained."""

    def __init__(self, dim, heads, dropout, scale, attention_dropout=0.):
        super().__init__()
        self.query = nn.Parameter(torch.randn(NUM_CAMERAS, dim)*.02)
        self.null = nn.Parameter(torch.zeros(1, dim))
        self.norm = nn.LayerNorm(dim)
        self.pool = Attention(dim, heads, attention_dropout)
        self.mix = Block(dim, heads, dropout, cross=False, attention_dropout=attention_dropout)
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
    """tokens [N,T,dim], token validity [N,T] (1/0), capture-to-query gaps [N,T] (TIME_UNIT_S),
    anchor features [N,42,config.anchor_features] -> offset [N,42,3] from the anchor; zero until trained.
    An invalid token's score is exactly MASK_OFF: the learned time bias cannot revive it.
    The gap sets how much a token is attended (time_bias) and, with gap_embedding, is added
    to the token itself so values carry their age; zero-initialised, so it starts inert."""

    def __init__(self, dim, heads, blocks, dropout, gap_embedding=True, anchor_features=9, attention_dropout=0.):
        super().__init__()
        self.joints = nn.Parameter(torch.randn(HAND_JOINTS, dim)*.02)
        self.anchor = nn.Sequential(nn.Linear(anchor_features, dim), nn.GELU(), nn.Linear(dim, dim))
        self.null = nn.Parameter(torch.zeros(1, dim))
        self.time_bias = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, heads))
        self.gap = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, dim)) if gap_embedding else None
        if self.gap is not None:
            nn.init.zeros_(self.gap[-1].weight)
            nn.init.zeros_(self.gap[-1].bias)
        self.blocks = nn.ModuleList([Block(dim, heads, dropout, attention_dropout=attention_dropout)
                                     for _ in range(blocks)])
        self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 3))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, tokens, token_valid, gaps, anchor_features):
        n, _, d = tokens.shape
        x = self.joints+self.anchor(anchor_features)
        if self.gap is not None:
            tokens = tokens+self.gap(gaps[..., None])*token_valid[..., None]                   # padding stays zero
        keys = torch.cat((self.null.expand(n, 1, d), tokens), 1)
        valid = token_valid[:, None]
        bias = self.time_bias(gaps[..., None]).permute(0, 2, 1)*valid+(valid-1.)*-MASK_OFF      # [N,h,T]
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
    tokens: torch.Tensor     # [B,E,event_tokens,dim]
    point: torch.Tensor      # [B,E,2,21,3] nominal triangulation of those rays
    ok: torch.Tensor         # [B,E,2,21]
    stamp: torch.Tensor      # [B,E,2,21] mean capture time of the rays used

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
        self.encoder = EventEncoder(c.dim, c.finger_tokens, JOINT_FEATURES)
        self.calibrator = Calibrator(c.dim, c.heads, c.dropout, scale, c.attention_dropout) if c.calibration_head else None
        self.decoder = Decoder(c.dim, c.heads, c.blocks, c.dropout, c.gap_embedding, c.anchor_features, c.attention_dropout)

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
            point, ok, stamp, _ = sample_points(origin, direction, mask, capture)
            raw, valid = events['raw'][:, targets], events['valid'][:, targets]
            features = joint_features(raw, valid, point, ok)
        camera = F.one_hot(events['camera'][:, targets], NUM_CAMERAS).float()
        tokens = self.encoder(features.reshape(b*t, NUM_HANDS, HAND_FEATURES), camera.reshape(b*t, NUM_CAMERAS))
        evident = (ok & valid).flatten(2).any(-1) & events['present'][:, targets]
        return LiteEncoded(origin, direction, mask, capture, evident, tokens.reshape(b, t, *tokens.shape[1:]),
                           point, ok & events['present'][:, targets, None, None], stamp)

    def select(self, events, query):
        """Slots [B,Q,3K] and validity (events.select_slots)."""
        return select_slots(events, query, self.config.slots_per_camera, self.config.event_span_s)

    def decode_queries(self, events, encoded, query):
        """Pose [B,Q,2,21,3], calibration [B,Q,3,6] and whether a query had any slot [B,Q]."""
        c = self.config
        b, q = query.shape
        n, s, per = b*q, self.slots, c.event_tokens
        slot, valid = self.select(events, query)                                           # [B,Q,S]
        take = lambda value: _gather(value, slot).reshape(n, s, *value.shape[2:])
        camera, capture, arrival = take(events['camera']), take(events['capture']), take(events['arrival'])
        valid = valid.reshape(n, s)
        token_valid = valid.repeat_interleave(per, -1)
        # Padding slots point at event 0: zero what they carry, as the deployment runtime does.
        tokens = torch.where(token_valid[..., None], take(encoded.tokens).reshape(n, per*s, -1), 0.)
        token_camera = camera.repeat_interleave(per, -1)
        gaps = torch.where(valid, ((query.reshape(n, 1)-capture)/TIME_UNIT_S).clamp_min(0.), 0.)
        gaps = gaps.repeat_interleave(per, -1)
        cameras = torch.arange(NUM_CAMERAS, device=query.device)[:, None]
        if self.calibrator is not None:
            pool = key_bias((token_camera[:, None] == cameras) & token_valid[:, None])        # [N,3,2S]
            params = self.calibrator(tokens, pool).float()
            # A camera without triangulated evidence in its slots keeps the nominal calibration.
            informative = ((camera[:, None] == cameras) & (valid & take(encoded.evident))[:, None]).any(-1)
            params = torch.where(informative[..., None], params, 0.)
        else:
            params = tokens.new_zeros(n, NUM_CAMERAS, CALIBRATION_PARAMS)
        # The anchor's geometry needs only the slots that can reach it: a sample is at most as
        # new as its event's arrival, so a slot that arrived over anchor_hold_s before the query
        # is never used. Slots are in arrival order (padding first), so these are the last ones.
        reach = self.reaching_slots(valid, arrival, query.reshape(n, 1))
        near = lambda value: value[:, s-reach:]
        near_valid = near(valid)
        with torch.autocast(device_type=query.device.type, enabled=False):
            point, ok, stamp, rays = sample_points(near(take(encoded.origin)), near(take(encoded.direction)),
                                                   near(take(encoded.mask)), near(take(encoded.capture)),
                                                   params[:, None].expand(n, reach, -1, -1),
                                                   outlier_ratio=c.ray_outlier_ratio)
            ok = ok & near_valid[..., None, None]
            single = (self.single_rays(events, encoded, query, rays, near(take(encoded.mask)), near(take(encoded.capture)),
                                       near_valid) if c.anchor_ray_depth else None)
            anchor, flags = query_anchor(point, ok, stamp, near(arrival), near_valid, query.reshape(n, 1),
                                         c.anchor_lookback_s, c.anchor_hold_s, single=single)
        features = torch.cat((anchor, flags), -1).reshape(n, HAND_JOINTS, c.anchor_features)
        offset = self.decoder(tokens, token_valid.to(tokens.dtype), gaps, features)
        pose = anchor.reshape(n, HAND_JOINTS, 3)+offset.float()
        return (pose.reshape(b, q, NUM_HANDS, NUM_JOINTS, 3), params.reshape(b, q, NUM_CAMERAS, CALIBRATION_PARAMS),
                valid.any(-1).reshape(b, q))

    def reaching_slots(self, valid, arrival, query):
        """How many trailing slots any query of the batch needs for its anchor (at least 1:
        the latest slot gives the anchor's 'now' flag and single rays)."""
        reach = valid & (arrival >= query-self.config.anchor_hold_s-REACH_MARGIN_S)
        return max(int(reach.sum(-1).max()), 1)

    def single_rays(self, events, encoded, query, rays, mask, capture, valid):
        """query_anchor's single-ray inputs [N,1,2,21,...]: each camera's (calibrated) ray of
        each joint in the latest slot event's ray table [N,1,2,21,3(,3)], and the joint's
        newest nominal triangulation among all events arrived by the query and captured
        within anchor_depth_memory_s."""
        n = valid.shape[0]
        joints_last = lambda value: value.movedim(1, 3)                                       # camera axis after joint
        origins, directions = joints_last(rays[0][:, -1]), joints_last(rays[1][:, -1])        # [N,2,21,3,3]
        mask = joints_last(mask[:, -1] & valid[:, -1, None, None, None])                     # [N,2,21,3]
        ray_capture = capture[:, -1, None, None, :].expand_as(mask)
        index = torch.arange(events['camera'].shape[1], device=query.device)
        arrived = events['present'][:, None] & (events['arrival'][:, None] <= query[..., None]+ARRIVAL_TOLERANCE_S)
        remembered = (arrived[..., None, None] & encoded.ok[:, None]
                      & (encoded.stamp[:, None] >= query[..., None, None, None]-self.config.anchor_depth_memory_s))
        latest = torch.where(remembered, index[:, None, None], -1).amax(2)                  # [B,Q,2,21]
        reference = _gather_joint(encoded.point, latest.clamp_min(0))
        per_query = (origins, directions, mask, ray_capture,
                     reference.reshape(n, NUM_HANDS, NUM_JOINTS, 3), (latest >= 0).reshape(n, NUM_HANDS, NUM_JOINTS))
        return tuple(value[:, None] for value in per_query)                                 # query_anchor's Q=1

    def forward(self, event_features, event_valid, event_camera, event_capture, event_arrival, event_present,
                query_times, return_details=False):
        """Model inputs (contracts.MODEL_INPUT_KEYS). return_details gives ModelOutput whose
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

    def to_absolute(self, encoded, origin):
        return replace(encoded, capture=encoded.capture.double()+origin, stamp=encoded.stamp.double()+origin)

    def decode(self, time):
        encoded = self.store.encoded.map(lambda value: value[None])
        encoded = replace(encoded, capture=(encoded.capture-time).float(), stamp=(encoded.stamp-time).float())
        query = torch.zeros(1, 1, device=self.device)
        return self.model.decode_queries(relative_events(self.store.raw, time), encoded, query)[0][0, 0]
