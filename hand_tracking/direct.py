"""HandDirect: camera events to the 3D joints of both hands with networks only.

  per event, once, on arrival (from that frame alone)
    features  u,v, ray origin and direction, capture->arrival delay, detected   [2,21,10]
    encoder   one token per finger of each hand                                [10, dim]
  per query at time t
    slots     each camera's slots_per_camera latest events captured within event_span_s
    fusion    self-attention over the slots' tokens, each told its age (t - capture)
    decoder   42 joint queries attend to the fused tokens -> xyz                  [42, 3]

Decoder output (config.wrist_relative): each hand's wrist in world coordinates and its other
20 joints relative to that wrist, from separate heads, so hand position (tens of cm) and hand
shape (mm to cm) do not share one output scale. Coarse to fine (config.refine): every decoder
block outputs a pose and the next block, told that pose, predicts only its correction; every
stage is supervised (ModelOutput.stages). Both end in the absolute pose [42,3]; with neither,
the decoder is the original HandDirect's (same parameter names, older checkpoints load).

No triangulation, anchor or camera correction: the networks learn how views and times
combine. forward() computes exactly what DirectStream computes event by event. The
networks-only comparison for HandLiteV3 (hand_tracking.model), whose event encoder it shares;
the master Pi app (hand_tracker_master) runs it.
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

# Wrist of each hand among the 42 joint queries (hand-major, wrist first).
WRISTS = tuple(hand*NUM_JOINTS for hand in range(NUM_HANDS))


def wrist_layout():
    """Constants of the wrist decomposition: wrist mask [42,1] (1 on the wrists) and compose
    [42,42], mapping (wrists absolute, other joints relative to their wrist) to absolute joints."""
    wrist = torch.zeros(HAND_JOINTS, 1)
    compose = torch.eye(HAND_JOINTS)
    for root in WRISTS:
        wrist[root] = 1.
        compose[root+1:root+NUM_JOINTS, root] = 1.
    return wrist, compose


def pose_head(dim, zero=False):
    """Joint tokens [...,dim] -> xyz [...,3]; zero: starts at zero output (a correction)."""
    head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 3))
    if zero:
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
    return head


class WristHead(nn.Module):
    """Joint tokens [N,42,dim] -> [N,42,3]: the wrist head's output on the two wrists (absolute),
    the joint head's elsewhere (relative to the wrist). Both heads run on every token and a
    constant mask picks (no Gather, for ncnn)."""

    def __init__(self, dim, zero=False):
        super().__init__()
        self.wrist, self.joints = pose_head(dim, zero), pose_head(dim, zero)
        self.register_buffer('wrist_mask', wrist_layout()[0], persistent=False)

    def forward(self, x):
        joints = self.joints(x)
        return joints+self.wrist_mask*(self.wrist(x)-joints)


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
    attend to the valid tokens, then to each other.

    wrist_relative: heads output wrists absolute and other joints relative to their wrist
    (the "layout"), composed to absolute joints. refine: a head after every block; block k>0
    first adds an embedding of the current layout (detached) to its joint tokens and its head
    (zero-initialised) outputs a correction added to the layout. Without either, one head
    after the last block (the original HandDirect, parameter names unchanged)."""

    def __init__(self, dim, heads, blocks, dropout, attention_dropout, wrist_relative=True, refine=True):
        super().__init__()
        self.wrist_relative, self.refine = wrist_relative, refine
        self.joints = nn.Parameter(torch.randn(HAND_JOINTS, dim)*.02)
        self.null = nn.Parameter(torch.zeros(1, dim))
        self.blocks = nn.ModuleList([Block(dim, heads, dropout, attention_dropout=attention_dropout)
                                     for _ in range(blocks)])
        head = WristHead if wrist_relative else pose_head
        if refine:
            self.heads = nn.ModuleList([head(dim, zero=k > 0) for k in range(blocks)])
            self.pose_embedding = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
            nn.init.zeros_(self.pose_embedding[-1].weight)
            nn.init.zeros_(self.pose_embedding[-1].bias)
        else:
            self.output = head(dim)
        if wrist_relative:
            self.register_buffer('compose', wrist_layout()[1], persistent=False)

    def absolute(self, layout):
        """Layout [N,42,3] -> absolute joints [N,42,3] (a constant MatMul, as in EventEncoder)."""
        return self.compose@layout if self.wrist_relative else layout

    def forward(self, tokens, valid, stages=False):
        """Joints [N,42,3]; stages: the list of every stage's joints, coarse to final."""
        n, _, d = tokens.shape
        keys = torch.cat((self.null.expand(n, 1, d), tokens), 1)                           # a query with no slot
        bias = torch.cat((torch.zeros_like(valid[:, :1]), (valid-1.)*-MASK_OFF), -1)[:, None, None]
        x = self.joints.expand(n, -1, -1)
        if not self.refine:
            for block in self.blocks:
                x = block(x, keys, bias)
            pose = self.absolute(self.output(x))
            return [pose] if stages else pose
        layout, poses = None, []
        for block, head in zip(self.blocks, self.heads):
            if layout is not None:
                x = x+self.pose_embedding(layout.detach())
            x = block(x, keys, bias)
            step = head(x)
            layout = step if layout is None else layout+step
            poses.append(self.absolute(layout))
        return poses if stages else poses[-1]


class QueryNetwork(nn.Module):
    """The per-query networks as one graph: fusion then decoder. The exported graph is
    forward(tokens, valid, ages): the final joints only."""

    def __init__(self, fusion, decoder):
        super().__init__()
        self.fusion, self.decoder = fusion, decoder

    def forward(self, tokens, valid, ages, stages=False):
        return self.decoder(self.fusion(tokens, valid, ages), valid, stages)


class HandDirect(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or DirectConfig()
        c = self.config
        self.encoder = EventEncoder(c.dim, True, DIRECT_JOINT_FEATURES)
        self.query_network = QueryNetwork(Fusion(c.dim, c.heads, c.fusion_blocks, c.dropout, c.attention_dropout),
                                          Decoder(c.dim, c.heads, c.blocks, c.dropout, c.attention_dropout,
                                                  c.wrist_relative, c.refine))

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
        """Pose [B,Q,2,21,3], None (no camera correction), whether a query had any slot [B,Q]
        and the earlier refinement stages' poses [S-1,B,Q,2,21,3] (None with one stage)."""
        b, q = query.shape
        n, s, per = b*q, self.slots, self.config.event_tokens
        slot, valid = select_slots(events, query, self.config.slots_per_camera, self.config.event_span_s)
        take = lambda value: _gather(value, slot).reshape(n, s, *value.shape[2:])
        valid = valid.reshape(n, s)
        token_valid = repeat_each(valid, per)
        tokens = torch.where(token_valid[..., None], take(encoded.tokens).reshape(n, per*s, -1), 0.)
        ages = torch.where(valid, ((query.reshape(n, 1)-take(events['capture']))/TIME_UNIT_S).clamp_min(0.), 0.)
        poses = self.query_network(tokens, token_valid.to(tokens.dtype), repeat_each(ages, per), stages=True)
        poses = [pose.float().reshape(b, q, NUM_HANDS, NUM_JOINTS, 3) for pose in poses]
        stages = torch.stack(poses[:-1]) if len(poses) > 1 else None
        return poses[-1], None, valid.any(-1).reshape(b, q), stages

    def forward(self, event_features, event_valid, event_camera, event_capture, event_arrival, event_present,
                query_times, return_details=False):
        """Model inputs (contracts.MODEL_INPUT_KEYS). return_details gives ModelOutput with no
        calibration, accepted marking queries with any slot [B,Q] and the refinement stages."""
        if event_features.ndim != 5 or event_features.shape[2:] != (NUM_HANDS, NUM_JOINTS, FEATURE_CHANNELS):
            raise ValueError('Expected event features [B,E,2,21,14]')
        if event_valid.shape != event_features.shape[:-1] or query_times.ndim != 2:
            raise ValueError('Invalid event/query dimensions')
        events = make_events(event_features, event_valid, event_camera, event_capture, event_arrival, event_present)
        output = ModelOutput(*self.decode_queries(events, self.encode_events(events), query_times.float()))
        return output if return_details else output.pose


class DirectStream(EventStream):
    """HandDirect event by event: each event's tokens are computed once; a query selects the
    slots and runs fusion and decoder. Equals forward()."""

    def decode(self, time):
        encoded = self.store.encoded.map(lambda value: value[None])
        query = torch.zeros(1, 1, device=self.device)
        return self.model.decode_queries(relative_events(self.store.raw, time), encoded, query)[0][0, 0]
