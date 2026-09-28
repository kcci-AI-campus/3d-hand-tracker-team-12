"""HandDirect: camera events to the 3D joints of both hands with networks only.

  per event, once, on arrival (from that frame alone)
    features  u,v, ray origin and direction, capture->arrival delay, detected   [2,21,10]
    encoder   one token per finger of each hand                                [10, dim]
  per query at time t
    slots     each camera's slots_per_camera latest events captured within event_span_s
    fusion    self-attention over the slots' tokens, each told its age (t - capture)
    decoder   42 joint queries attend to the fused tokens -> xyz                  [42, 3]

No triangulation, anchor or camera correction: the networks learn how views and times
combine. forward() computes exactly what DirectStream computes event by event. Kept as the
networks-only comparison for HandLiteV3 (hand_tracking.model), whose event encoder it shares.
"""
import torch
from torch import nn
from .config import DirectConfig
from .constants import NUM_CAMERAS, NUM_HANDS, NUM_JOINTS, HAND_JOINTS, FEATURE_CHANNELS, TIME_UNIT_S, MASK_OFF
from .contracts import ModelOutput
from .events import _gather, make_events, relative_events, select_slots
from .layers import Block, EventEncoder
from .networks import DIRECT_JOINT_FEATURES, DirectEncoded, camera_one_hot, direct_features, repeat_each
from .stream import EventStream


class Fusion(nn.Module):
    """tokens [N,T,dim], validity [N,T] (1/0), ages [N,T] (TIME_UNIT_S) -> fused tokens [N,T,dim].
    Each token first adds an embedding of its age; padding tokens stay zero and are masked."""

    def __init__(self, dim, heads, blocks, dropout, attention_dropout):
        super().__init__()
        self.age = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, dim))
        self.blocks = nn.ModuleList([Block(dim, heads, dropout, cross=False, attention_dropout=attention_dropout)
                                     for _ in range(blocks)])

    def forward(self, tokens, valid, ages):
        x = (tokens+self.age(ages[..., None]))*valid[..., None]
        bias = ((valid-1.)*-MASK_OFF)[:, None, None]                                         # [N,1,1,T]
        for block in self.blocks:
            x = block(x, self_bias=bias)*valid[..., None]
        return x


class Decoder(nn.Module):
    """fused tokens [N,T,dim], validity [N,T] (1/0) -> joints [N,42,3]: learned joint queries
    attend to the valid tokens, then to each other."""

    def __init__(self, dim, heads, blocks, dropout, attention_dropout):
        super().__init__()
        self.joints = nn.Parameter(torch.randn(HAND_JOINTS, dim)*.02)
        self.null = nn.Parameter(torch.zeros(1, dim))
        self.blocks = nn.ModuleList([Block(dim, heads, dropout, attention_dropout=attention_dropout)
                                     for _ in range(blocks)])
        self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 3))

    def forward(self, tokens, valid):
        n, _, d = tokens.shape
        keys = torch.cat((self.null.expand(n, 1, d), tokens), 1)                           # a query with no slot
        bias = torch.cat((torch.zeros_like(valid[:, :1]), (valid-1.)*-MASK_OFF), -1)[:, None, None]
        x = self.joints.expand(n, -1, -1)
        for block in self.blocks:
            x = block(x, keys, bias)
        return self.output(x)


class QueryNetwork(nn.Module):
    """The per-query networks as one graph: fusion then decoder (see lite export)."""

    def __init__(self, fusion, decoder):
        super().__init__()
        self.fusion, self.decoder = fusion, decoder

    def forward(self, tokens, valid, ages):
        return self.decoder(self.fusion(tokens, valid, ages), valid)


class HandDirect(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or DirectConfig()
        c = self.config
        self.encoder = EventEncoder(c.dim, True, DIRECT_JOINT_FEATURES)
        self.query_network = QueryNetwork(Fusion(c.dim, c.heads, c.fusion_blocks, c.dropout, c.attention_dropout),
                                          Decoder(c.dim, c.heads, c.blocks, c.dropout, c.attention_dropout))

    @property
    def slots(self):
        return NUM_CAMERAS*self.config.slots_per_camera

    def encode_events(self, events, targets=None):
        """DirectEncoded of the target events (default: all); each from its own frame only."""
        b, e = events['camera'].shape
        if targets is None:
            targets = torch.arange(e, device=events['camera'].device)
        t = targets.numel()
        features = direct_features(events['raw'][:, targets], events['valid'][:, targets])
        camera = camera_one_hot(events['camera'][:, targets])
        tokens = self.encoder(features.reshape(b*t, NUM_HANDS, -1), camera.reshape(b*t, NUM_CAMERAS))
        return DirectEncoded(tokens.reshape(b, t, *tokens.shape[1:]))

    def decode_queries(self, events, encoded, query):
        """Pose [B,Q,2,21,3], None (no camera correction) and whether a query had any slot [B,Q]."""
        b, q = query.shape
        n, s, per = b*q, self.slots, self.config.event_tokens
        slot, valid = select_slots(events, query, self.config.slots_per_camera, self.config.event_span_s)
        take = lambda value: _gather(value, slot).reshape(n, s, *value.shape[2:])
        valid = valid.reshape(n, s)
        token_valid = repeat_each(valid, per)
        tokens = torch.where(token_valid[..., None], take(encoded.tokens).reshape(n, per*s, -1), 0.)
        ages = torch.where(valid, ((query.reshape(n, 1)-take(events['capture']))/TIME_UNIT_S).clamp_min(0.), 0.)
        pose = self.query_network(tokens, token_valid.to(tokens.dtype), repeat_each(ages, per))
        return pose.float().reshape(b, q, NUM_HANDS, NUM_JOINTS, 3), None, valid.any(-1).reshape(b, q)

    def forward(self, event_features, event_valid, event_camera, event_capture, event_arrival, event_present,
                query_times, return_details=False):
        """Model inputs (contracts.MODEL_INPUT_KEYS). return_details gives ModelOutput with no
        calibration and accepted marking queries with any slot [B,Q]."""
        if event_features.ndim != 5 or event_features.shape[2:] != (NUM_HANDS, NUM_JOINTS, FEATURE_CHANNELS):
            raise ValueError('Expected event features [B,E,2,21,14]')
        if event_valid.shape != event_features.shape[:-1] or query_times.ndim != 2:
            raise ValueError('Invalid event/query dimensions')
        events = make_events(event_features, event_valid, event_camera, event_capture, event_arrival, event_present)
        pose, calibration, rows = self.decode_queries(events, self.encode_events(events), query_times.float())
        return ModelOutput(pose, calibration, rows) if return_details else pose


class DirectStream(EventStream):
    """HandDirect event by event: each event's tokens are computed once; a query selects the
    slots and runs fusion and decoder. Equals forward()."""

    def decode(self, time):
        encoded = self.store.encoded.map(lambda value: value[None])
        query = torch.zeros(1, 1, device=self.device)
        return self.model.decode_queries(relative_events(self.store.raw, time), encoded, query)[0][0, 0]
