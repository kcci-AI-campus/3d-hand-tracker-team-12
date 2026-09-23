"""HandTransformer: learned per-event calibration, event tokens and the query decoder
around the model-free anchor of hand_tracking.events."""
import torch
import math
from torch import nn
import torch.nn.functional as F
from .config import ModelConfig
from .constants import (NUM_CAMERAS, NUM_HANDS, NUM_JOINTS, HAND_JOINTS, FEATURE_CHANNELS, UV, RAY_ORIGIN,
                        RAY_DIRECTION, DELAY, TIME_UNIT_S, CALIBRATION_PARAMS, ARRIVAL_TOLERANCE_S)
from .contracts import EncodedEvents, Evidence, LayerKV, ModelOutput, to_joint_sequences, to_pose_sequences
from .events import make_events, _up_to, latest_rays, sample_points, _own, _miss, query_anchor, _event_motion

# Per-joint ray features: u,v (2), ray origin (3) and direction (3), delay (1), miss vector (3).
RAY_FEATURES = 12
# Anchor position (3) and query_anchor flags (5).
ANCHOR_FEATURES = 8


class CrossAttention(nn.Module):
    """Multi-head attention of queries over keys with a boolean allow-mask and optional
    additive per-head bias (explicit; avoids the NaN-prone fast path of
    nn.TransformerEncoderLayer with per-batch float masks)."""
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads; self.dropout = dropout
        self.query = nn.Linear(dim, dim); self.key_value = nn.Linear(dim, 2*dim); self.out = nn.Linear(dim, dim)

    def project_keys(self, keys):
        """keys [...,d] -> key, value [...,h,d/h]; independent of the queries, so cacheable."""
        k, v = self.key_value(keys).reshape(*keys.shape[:-1], 2, self.heads, -1).unbind(-3)
        return k, v

    def attend(self, x, k, v, allowed, bias=None, null=None):
        """x [N,L,d], k/v [N,h,S,d/h], allowed [N,L,S], bias [N,h,L,S].

        null = projected (key, value) [h,d/h] of a learned token that every query may
        attend to with zero bias, so rows without any allowed key stay defined. Without
        it every row of allowed needs one True.
        """
        n, l, d = x.shape; h = self.heads
        if null is not None:
            k, v = (torch.cat((extra[None,:,None].expand(n,-1,-1,-1).to(kv.dtype), kv), 2) for extra, kv in zip(null, (k, v)))
            allowed = torch.cat((allowed.new_ones(n, l, 1), allowed), -1)
            if bias is not None: bias = F.pad(bias, (1, 0))
        q = self.query(x).reshape(n, l, h, d//h).transpose(1,2)
        # Written out rather than F.scaled_dot_product_attention: its export decomposition
        # breaks ONNX conversion, and these sizes (<=43 keys per row) gain nothing from it.
        scores = (q @ k.to(q.dtype).transpose(-1,-2))*(d//h)**-.5
        scores = scores.masked_fill(~allowed[:,None], float('-inf'))
        if bias is not None: scores = scores+bias.to(q.dtype)
        weights = F.dropout(scores.softmax(-1), self.dropout, self.training)
        out = weights @ v.to(q.dtype)
        return self.out(out.transpose(1,2).reshape(n, l, d))

    def forward(self, x, keys, allowed, bias=None, null=None):
        """x [N,L,d], keys [N,S,d], null token [d] or None -> [N,L,d]."""
        k, v = self.project_keys(keys)
        return self.attend(x, k.transpose(1,2), v.transpose(1,2), allowed, bias,
                           None if null is None else self.project_keys(null))


class DecoderLayer(nn.Module):
    """Pre-norm cross-attention of joint queries to event tokens, then FFN."""
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.norm_query = nn.LayerNorm(dim); self.norm_key = nn.LayerNorm(dim); self.norm_ff = nn.LayerNorm(dim)
        self.attention = CrossAttention(dim, heads, dropout)
        self.feedforward = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*dim, dim))
        self.dropout = nn.Dropout(dropout)

    def key_values(self, tokens):
        """Event tokens [...,d] -> this layer's key, value [...,h,d/h] (computed once per event)."""
        return self.attention.project_keys(self.norm_key(tokens))

    def forward(self, x, k, v, allowed, bias, null_token):
        null = self.key_values(null_token)
        x = x+self.dropout(self.attention.attend(self.norm_query(x), k, v, allowed, bias, null))
        return x+self.dropout(self.feedforward(self.norm_ff(x)))


class HandTransformer(nn.Module):
    """Pose at arbitrary query times from asynchronous camera events.

    Each event (one camera frame, 42 joint rays) gets, once, when it arrives:
      1. calibration evidence: its nominal rays against a triangulation with every
         camera's latest ray (frame-local, calibration-independent);
      2. a learned per-camera calibration from the evidence of events that arrived within
         calibration_span_s before it (no long-term memory);
      3. a calibrated triangulation sample and an event token (42 joints, one camera).
    A query at time t anchors each joint on the samples of events arrived by t (line fit
    extrapolated to t), then joint queries attend to the tokens of events captured within
    event_span_s (bias from the capture-to-query gap) and to each other; the output is the
    anchor plus a learned offset. Nothing depends on frame counts or the query rate.

    encode_events() (steps 1-3) and decode_queries() are the two halves of forward();
    EventStream calls them with per-event caching and gets results identical to forward().
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        def mlp(inputs):
            return nn.Sequential(nn.Linear(inputs, c.dim), nn.GELU(), nn.Linear(c.dim, c.dim))
        def layer():
            return nn.TransformerEncoderLayer(c.dim, c.heads, 4*c.dim, c.dropout,
                                               activation='gelu', batch_first=True, norm_first=True)
        self.input_projection = mlp(RAY_FEATURES+CALIBRATION_PARAMS)
        self.anchor_projection = mlp(ANCHOR_FEATURES)
        self.camera_embedding = nn.Embedding(NUM_CAMERAS, c.dim)
        self.hand_embedding = nn.Embedding(NUM_HANDS, c.dim)
        self.joint_embedding = nn.Embedding(NUM_JOINTS, c.dim)
        self.query_embedding = nn.Parameter(torch.zeros(c.dim))
        self.event_null = nn.Parameter(torch.zeros(c.dim))
        # Per-event spatial encoder over the 42 joints seen by one camera.
        self.encoder = nn.ModuleList([layer() for _ in range(c.encoder_layers)])
        # Attention bias per head from the capture-to-query gap (seconds, no frame index).
        self.time_bias = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, c.heads))
        self.cross = nn.ModuleList([DecoderLayer(c.dim, c.heads, c.dropout) for _ in range(c.blocks)])
        self.spatial = nn.ModuleList([layer() for _ in range(c.blocks)])
        self.output = nn.Sequential(nn.LayerNorm(c.dim), nn.Linear(c.dim, c.dim), nn.GELU(), nn.Linear(c.dim, 3))
        # Zero residual heads: an untrained model reproduces the uncalibrated anchor.
        nn.init.zeros_(self.output[-1].weight); nn.init.zeros_(self.output[-1].bias)
        if c.calibration_head:
            self.evidence_projection = mlp(RAY_FEATURES)
            self.calibration_query = nn.Parameter(torch.zeros(NUM_CAMERAS, c.dim))
            self.calibration_null = nn.Parameter(torch.zeros(c.dim))
            self.calibration_pool = CrossAttention(c.dim, c.heads, c.dropout)
            self.calibration_mix = layer()
            self.calibration_output = nn.Sequential(nn.LayerNorm(c.dim), nn.Linear(c.dim, c.dim), nn.GELU(),
                                                    nn.Linear(c.dim, CALIBRATION_PARAMS))
            nn.init.zeros_(self.calibration_output[-1].weight); nn.init.zeros_(self.calibration_output[-1].bias)
        self.register_buffer('calibration_scale', torch.tensor(
            [math.radians(c.calibration_rotation_deg)]*3+[c.calibration_position]*3))

    def _joints(self, device):
        return (self.hand_embedding(torch.arange(NUM_HANDS, device=device))[:,None] +
                self.joint_embedding(torch.arange(NUM_JOINTS, device=device))[None]).reshape(HAND_JOINTS, self.config.dim)

    @staticmethod
    def _ray_features(raw, origin, direction, miss):
        """u,v in [-1,1], ray origin/direction, capture->arrival delay (TIME_UNIT_S) and miss vector."""
        return torch.cat((raw[...,UV]*2-1, origin, direction, raw[...,DELAY]/TIME_UNIT_S, miss), -1)

    def _evidence(self, events, targets, rays):
        """Per-event calibration evidence summary [B,T,d] and whether it has any [B,T]."""
        c = self.config; b = events['camera'].shape[0]; t = targets.numel(); d = c.dim
        with torch.no_grad(), torch.autocast(device_type=events['raw'].device.type, enabled=False):
            point, ok, _, _ = sample_points(*rays)
            raw = events['raw'][:,targets]; own = events['valid'][:,targets]
            origin, direction = raw[...,RAY_ORIGIN], raw[...,RAY_DIRECTION]
            miss = _miss(point, ok, origin, direction, own)
            x = self._ray_features(raw, origin, direction, miss)
        use = own & ok
        x = self.evidence_projection(torch.where(own[...,None], x, 0.)).reshape(b, t, HAND_JOINTS, d)
        x = x+self.camera_embedding(events['camera'][:,targets])[:,:,None]+self._joints(x.device)
        weight = use.reshape(b, t, HAND_JOINTS, 1).float()
        return Evidence((x*weight).sum(2)/weight.sum(2).clamp_min(1), use.flatten(2).any(-1))

    def _calibrate(self, events, evidence, targets):
        """Calibration [B,T,3,6] at each target event from evidence that arrived within span."""
        c = self.config; b, e = events['camera'].shape; t = targets.numel(); d = c.dim; h = c.heads
        arrival = events['arrival']; target_arrival = arrival[:,targets]
        cameras = torch.arange(NUM_CAMERAS, device=arrival.device)
        band = _up_to(events, targets) & (arrival[:,None] > target_arrival[...,None]-c.calibration_span_s)   # [B,T,E]
        allowed = band[:,:,None] & (events['camera'][:,None,None] == cameras[:,None])     # [B,T,3,E]
        # Every camera's query pools over the same evidence keys: project them once.
        k, v = (value.transpose(1,2)[:,None].expand(b, NUM_CAMERAS, h, e, d//h).reshape(b*NUM_CAMERAS, h, e, d//h)
                for value in self.calibration_pool.project_keys(evidence.summary))
        mask = allowed.permute(0,2,1,3).reshape(b*NUM_CAMERAS, t, e)
        query = (self.calibration_query+self.camera_embedding.weight)[None,:,None].expand(b,NUM_CAMERAS,t,d)
        pooled = self.calibration_pool.attend(query.reshape(b*NUM_CAMERAS,t,d), k, v, mask,
                                              null=self.calibration_pool.project_keys(self.calibration_null))
        pooled = pooled.reshape(b,NUM_CAMERAS,t,d).permute(0,2,1,3).reshape(b*t,NUM_CAMERAS,d)
        correction = self.calibration_output(self.calibration_mix(pooled)).float().reshape(b,t,NUM_CAMERAS,CALIBRATION_PARAMS)
        # A camera with no evidence (fresh ray seen with another camera) keeps the nominal calibration.
        informative = (allowed & evidence.evident[:,None,None]).any(-1)
        return torch.where(informative[...,None], correction*self.calibration_scale, 0.)

    def _encode(self, events, params, targets, rays):
        """Calibrated sample (point, ok, stamp), own calibrated rays and event tokens [B,T,42,d]."""
        c = self.config; b = events['camera'].shape[0]; t = targets.numel(); d = c.dim
        camera = events['camera'][:,targets]
        with torch.autocast(device_type=events['raw'].device.type, enabled=False):
            point, ok, stamp, (co, cd) = sample_points(*rays, params.float(), c.anchor_ray_gate)
            raw = events['raw'][:,targets]; own = events['valid'][:,targets]
            own_origin, own_direction = _own(co, camera), _own(cd, camera)
            miss = _miss(point, ok, own_origin, own_direction, own)
            correction = (_own(params, camera)/self.calibration_scale)[:,:,None,None]
            correction = correction.expand(b, t, NUM_HANDS, NUM_JOINTS, CALIBRATION_PARAMS)
            x = torch.cat((self._ray_features(raw, own_origin, own_direction, miss), correction), -1)
        x = self.input_projection(torch.where(own[...,None], x, 0.)).reshape(b, t, HAND_JOINTS, d)
        x = x+self.camera_embedding(camera)[:,:,None]+self._joints(x.device)
        x = x.reshape(b*t, HAND_JOINTS, d)
        for layer in self.encoder: x = layer(x)
        ok = ok & events['present'][:,targets,None,None]
        tokens = x.reshape(b, t, HAND_JOINTS, d)
        # Decoder keys/values depend only on the event: cache them with it.
        keys = tuple(LayerKV(*layer.key_values(tokens)) for layer in self.cross)
        return EncodedEvents(point, ok, stamp, keys, own_origin, own_direction)

    def encode_events(self, events, targets=None, previous_evidence=None):
        """Encode target events (default: all) given every earlier event in `events`.

        Returns (EncodedEvents, calibration [B,T,3,6], evidence of the targets or None).
        previous_evidence covers the events before the targets (a stream's cached history);
        without it the targets are the whole history. Nothing is stored on the module.
        """
        if targets is None:
            targets = torch.arange(events['camera'].shape[1], device=events['camera'].device)
        with torch.autocast(device_type=events['raw'].device.type, enabled=False):
            rays = latest_rays(events, targets, self.config.sample_max_age_s)
        evidence = None
        if self.config.calibration_head:
            evidence = self._evidence(events, targets, rays)
            history = evidence if previous_evidence is None else previous_evidence.append(evidence)
            params = self._calibrate(events, history, targets)
        else:
            params = events['raw'].new_zeros(events['camera'].shape[0], targets.numel(), NUM_CAMERAS, CALIBRATION_PARAMS)
        return self._encode(events, params, targets, rays), params, evidence

    def decode_queries(self, events, encoded, query):
        """Pose [B,Q,2,21,3] at query times [B,Q] (same time origin as the events)."""
        c = self.config; b, e = events['camera'].shape; q = query.shape[1]; d = c.dim; h = c.heads
        with torch.autocast(device_type=query.device.type, enabled=False):
            motion = None
            if c.anchor_motion_fit and c.anchor_lookback_s > 0:
                motion = _event_motion(events, encoded.origin, encoded.direction, query, c.anchor_lookback_s)
            anchor, flags = query_anchor(encoded.point, encoded.ok, encoded.stamp, events['arrival'],
                                         events['present'], query, c.anchor_lookback_s, c.anchor_hold_s, motion)
        x = torch.cat((anchor, flags), -1).reshape(b, q, HAND_JOINTS, ANCHOR_FEATURES)
        x = self._joints(query.device)+self.anchor_projection(x)+self.query_embedding
        gap = query[:,:,None]-events['capture'][:,None]                                      # [B,Q,E]
        allowed = (events['present'][:,None] & (events['arrival'][:,None] <= query[...,None]+ARRIVAL_TOLERANCE_S)
                   & (gap <= c.event_span_s))
        allowed = allowed[...,None] & events['valid'].reshape(b, 1, e, HAND_JOINTS)          # [B,Q,E,42]
        allowed = allowed.permute(0,3,1,2).reshape(b*HAND_JOINTS, q, e)
        bias = self.time_bias(gap.clamp_min(0)[...,None]/TIME_UNIT_S).permute(0,3,1,2)      # [B,h,Q,E]
        bias = bias[:,None].expand(b, HAND_JOINTS, h, q, e).reshape(b*HAND_JOINTS, h, q, e)
        x = to_joint_sequences(x)
        for cross, spatial, kv in zip(self.cross, self.spatial, encoded.keys, strict=True):
            k, v = (value.permute(0,2,3,1,4).reshape(b*HAND_JOINTS, h, e, d//h) for value in kv)
            x = cross(x, k, v, allowed, bias, self.event_null)
            x = spatial(to_pose_sequences(x, b).reshape(b*q, HAND_JOINTS, d))
            x = to_joint_sequences(x.reshape(b, q, HAND_JOINTS, d))
        x = to_pose_sequences(x, b)
        return anchor+self.output(x).reshape(b, q, NUM_HANDS, NUM_JOINTS, 3).float()

    def forward(self, event_features, event_valid, event_camera, event_capture, event_arrival, event_present,
                query_times, return_details=False):
        """Events [B,E,2,21,14] (u,v · ray origin/direction · times · delay · ids), valid
        [B,E,2,21], camera [B,E], capture/arrival [B,E] seconds, present [B,E], sorted by
        arrival (ties: input order is arrival order); query_times [B,Q] seconds on the same
        origin -> pose [B,Q,2,21,3].

        Padding and captures no newer than that camera's prior maximum are ignored.
        A query uses only events that arrived by its time. return_details gives a
        ModelOutput that adds each event's calibration [B,E,3,6] (rotation vector rad,
        translation world units; see apply_calibration) and which events were accepted.
        No actual calibration or ground truth is used as input.
        """
        if event_features.ndim != 5 or event_features.shape[2:] != (NUM_HANDS, NUM_JOINTS, FEATURE_CHANNELS):
            raise ValueError('Expected event features [B,E,2,21,14]')
        if event_valid.shape != event_features.shape[:-1] or query_times.ndim != 2:
            raise ValueError('Invalid event/query dimensions')
        events = make_events(event_features, event_valid, event_camera, event_capture, event_arrival, event_present)
        encoded, params, _ = self.encode_events(events)
        pose = self.decode_queries(events, encoded, query_times.float())
        return ModelOutput(pose, params, events['present']) if return_details else pose
