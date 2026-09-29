"""HandLiteV3: a line-fit triangulation as anchor plus a learned correction, and each joint's
expected error. No camera calibration head and no state carried between queries.

  per event, once, on arrival
    features  u,v, ray origin and direction, capture->arrival delay, detected   [2,21,10]
    encoder   one token per finger of each hand                                [10, dim]
  per query at time t (geometry, no learned parameters, float32)
    rays      per camera and joint, a least-squares line over time through the ray directions
              of the detections within fit_span_s, read at t (one detection, or detections
              too close in time: their mean)
    prior     each joint's own newest triangulation before t: the same line fit and
              triangulation at t - k*prior_step_s (k = 1 .. prior_steps, fit windows that do not
              overlap), the newest that succeeded (observations only: no state, no feedback)
    anchor    triangulated: the least-squares point of the joint's rays. A joint that fails
              now is anchored from its prior point: on the ray (of the cameras that see it)
              passing closest to it, at the ray's point nearest to it (the direction observed
              now, the depth from the past); seeing none, at the prior point; without a prior
              point: zero                                                      [42, 3]
    joints    per joint: anchor, anchor minus prior point, how it was anchored, the prior
              point's age; per camera: detections, fitted, newest/oldest age, newest u,v, the
              ray's miss to the anchor, the fit's shift from the newest ray (an overreaching
              line), the detections' RMS and the newest one's residual from the line (a
              jumping detection)                                                 [42, 56]
  per query (networks)
    slots     each camera's slots_per_camera latest events within event_span_s, their
              tokens told their age (t - capture)                               [10*3K, dim]
    corrector 42 joint tokens (MLP of the joint features + a learned joint embedding):
              [cross-attention to the slot tokens, self-attention over the joints of both
              hands, feed-forward] x blocks -> offset (zero-initialised head)    [42, 3]
    pose      anchor + offset (a joint with no anchor: the offset alone)
    error     (config.error_estimate) per joint, from its final token (detached): its own
              expected error, log(1 + mm), trained on the actual error          [42]

State (LiteV3Stream and the deployment runtime only): each joint's last triangulation however
old, updated whenever it triangulates, and used as the prior point only when the history
search's, so a joint seen by one camera keeps its depth; its age input is capped at
max_prior_age_s. Training (forward()) has no state: its history search reaches prior_span_s back.
Nothing learned is fed back.
"""
import contextlib
import torch
from torch import nn
from .config import LiteV3Config
from .constants import (NUM_CAMERAS, NUM_HANDS, NUM_JOINTS, HAND_JOINTS, FINGERS, FEATURE_CHANNELS, UV, RAY_ORIGIN,
                        RAY_DIRECTION, TIME_UNIT_S, MISS_SCALE, RESIDUAL_UNIT, ARRIVAL_TOLERANCE_S, MIN_TIME_STD_S,
                        MASK_OFF)
from .contracts import ModelOutput
from .events import _gather, make_events, relative_events, select_slots
from .geometry import ray_residuals, triangulate
from .layers import Block, EventEncoder
from .networks import DIRECT_JOINT_FEATURES, DirectEncoded, camera_one_hot, direct_features, repeat_each
from .stream import EventStream

# Per camera: detected, fitted (a slope used), detections / 4, newest and oldest detection age
# (TIME_UNIT_S), newest u,v in [-1,1] (2), the ray's miss vector to the anchor (3) and the fitted
# ray's shift from the newest detection's ray (3), both x MISS_SCALE, the detections' RMS and the
# newest one's angular residual from the line (RESIDUAL_UNIT). Zero for a camera without a detection.
CAMERA_FEATURES = 15
# Per joint: anchor (3), anchor minus its own prior point (3, zero without one), anchored by
# triangulation / on one ray at the prior point's depth / at the prior point (3 flags), a prior
# point exists, the prior point's age (TIME_UNIT_S).
JOINT_FEATURES = 11
LITE3_FEATURES = JOINT_FEATURES+NUM_CAMERAS*CAMERA_FEATURES
# Encoder tokens per event: one per finger of each hand.
EVENT_TOKENS = NUM_HANDS*FINGERS
# ModelOutput.anchored values: no anchor, triangulated, one ray at the prior point's depth, the prior point.
ANCHOR_KINDS = ('none', 'triangulated', 'ray', 'prior')
FAR = 1e6   # a capture time no event has (finite, for XLA)


def float32(device):
    """A region computed in float32 even under CUDA/CPU autocast (XLA trains in float32 anyway)."""
    return torch.autocast(device_type=device.type, enabled=False) if device.type in ('cuda', 'cpu') else contextlib.nullcontext()


def gather_joints(values, pick):
    """values [B,E,42,C], pick [B,...,42] (event index per joint) -> [B,...,42,C]."""
    b, e, joints, c = values.shape
    flat = pick.reshape(b, -1, joints).permute(0, 2, 1)                                     # [B,42,M]
    out = values.permute(0, 2, 1, 3).gather(2, flat[..., None].expand(-1, -1, -1, c))       # [B,42,M,C]
    return out.permute(0, 2, 1, 3).reshape(*pick.shape, c)


def fitted_rays(events, query, span_s, offset_s=0.):
    """Each camera's ray per joint at the reference time query - offset_s, from a least-squares
    line over time through the ray directions of that camera's events in which the joint was
    detected, arrived by the query and captured within span_s before the reference time (not
    after it), read at the reference time (one detection, or detections too close in time:
    their mean). Returns per camera and joint [B,Q,3,42,...]: origin (3, mean), unit direction
    (3), detected, fitted (a slope used), detections, newest and oldest detection age from the
    reference time (s), the newest detection's u,v (2) and unit direction (3), and the
    detections' RMS and the newest one's angular residual from the line (radians)."""
    b, e = events['camera'].shape
    index = torch.arange(e, device=query.device)
    reference = query-offset_s
    usable = (events['present'][:, None] & (events['arrival'][:, None] <= query[..., None]+ARRIVAL_TOLERANCE_S)
              & (events['capture'][:, None] >= reference[..., None]-span_s)
              & (events['capture'][:, None] <= reference[..., None]+ARRIVAL_TOLERANCE_S))     # [B,Q,E]
    detected = events['valid'].reshape(b, 1, e, HAND_JOINTS)
    tau = (events['capture'][:, None]-reference[..., None])[..., None]                      # [B,Q,E,1]
    raw = events['raw'].reshape(b, e, HAND_JOINTS, -1)
    origin, direction = raw[..., RAY_ORIGIN], raw[..., RAY_DIRECTION]                        # [B,E,42,3]
    square = direction.square().sum(-1)                                                     # [B,E,42]
    unit = lambda d: d/d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    rays = []
    for camera in range(NUM_CAMERAS):
        use = (usable & (events['camera'][:, None] == camera))[..., None] & detected        # [B,Q,E,42]
        m = use.float()
        mt = m*tau
        s0, s1, s2 = m.sum(2), mt.sum(2), (mt*tau).sum(2)                                   # [B,Q,42]
        sd = torch.einsum('bqej,bejc->bqjc', m, direction)
        sdt = torch.einsum('bqej,bejc->bqjc', mt, direction)
        spread = s0*s2-s1*s1                                                                # s0^2 x time variance
        fit = (s0 >= 2) & (spread > MIN_TIME_STD_S**2*s0*s0)
        slope = torch.where(fit[..., None], (s0[..., None]*sdt-s1[..., None]*sd)/spread.clamp_min(1e-12)[..., None], 0.)
        value = (sd-slope*s1[..., None])/s0.clamp_min(1)[..., None]                          # the line at tau = 0
        mean_origin = torch.einsum('bqej,bejc->bqjc', m, origin)/s0.clamp_min(1)[..., None]
        # Sum over detections of |d - value - slope*tau|^2, expanded into the sums above.
        squared = (torch.einsum('bqej,bej->bqj', m, square)-2*(value*sd).sum(-1)-2*(slope*sdt).sum(-1)
                   +value.square().sum(-1)*s0+2*(value*slope).sum(-1)*s1+slope.square().sum(-1)*s2).clamp_min(0.)
        scale = value.norm(dim=-1).clamp_min(1e-8)                                          # residuals as angles
        rms = (squared/s0.clamp_min(1)).sqrt()/scale
        has = use.any(2)
        newest_index = torch.where(use, index[:, None], -1).amax(2).clamp_min(0)            # [B,Q,42]
        newest = gather_joints(raw, newest_index)                                           # [B,Q,42,11]
        newest_tau = torch.where(use, tau, -FAR).amax(2)
        oldest_tau = torch.where(use, tau, FAR).amin(2)
        newest_residual = (newest[..., RAY_DIRECTION]-value-slope*newest_tau.clamp_min(-span_s)[..., None]).norm(dim=-1)/scale
        age = lambda value: torch.where(has, -value, 0.)
        rays.append((mean_origin, unit(value), has, fit, s0, age(newest_tau), age(oldest_tau), newest[..., UV],
                     unit(newest[..., RAY_DIRECTION]), torch.where(has, rms, 0.), torch.where(has, newest_residual, 0.)))
    return tuple(torch.stack(values, 2) for values in zip(*rays))


def rays_at(events, query, config, offset_s=0.):
    """fitted_rays at query - offset_s, per joint and camera [B,Q,2,21,3,...]."""
    b, q = query.shape
    return tuple(value.transpose(2, 3).reshape(b, q, NUM_HANDS, NUM_JOINTS, NUM_CAMERAS, *value.shape[4:])
                 for value in fitted_rays(events, query, config.fit_span_s, offset_s))


def prior_points(events, query, config):
    """Each joint's own newest triangulation before the query: the same line fit and
    triangulation at query - k*prior_step_s (k = 1 .. prior_steps, newest first) -> point
    [B,Q,2,21,3], found [B,Q,2,21] and the age of the newest detection it used (s) [B,Q,2,21]."""
    b, q = query.shape
    point = torch.zeros(b, q, NUM_HANDS, NUM_JOINTS, 3, device=query.device)
    found = torch.zeros(b, q, NUM_HANDS, NUM_JOINTS, dtype=torch.bool, device=query.device)
    age = torch.zeros(b, q, NUM_HANDS, NUM_JOINTS, device=query.device)
    for k in range(1, config.prior_steps+1):
        offset = k*config.prior_step_s
        origin, direction, has, _, _, newest_age = rays_at(events, query, config, offset)[:6]
        p, ok = triangulate(origin, direction, has)
        take = ok & ~found
        point = torch.where(take[..., None], p, point)
        age = torch.where(take, torch.where(has, newest_age, FAR).amin(-1)+offset, age)
        found = found | ok
    return point, found, age


def merge_priors(searched, stored):
    """The history search's prior point per joint, or the stored one where the search found none:
    each (point [...,3], found [...], age (s) [...]). A newer stored point is not preferred, so a
    stream's inputs equal forward()'s (training's) wherever the search finds a point."""
    (point, found, age), (kept, known, kept_age) = searched, stored
    use = known & ~found
    return (torch.where(use[..., None], kept, point), found | known, torch.where(use, kept_age, age))


def anchor_features(events, query, config, stored=None):
    """Anchor [B,Q,2,21,3], how each joint was anchored (ANCHOR_KINDS index) [B,Q,2,21], the
    corrector's joint input [B,Q,42,F] and the age of the newest detection each triangulated
    joint used (s) [B,Q,2,21] (a stream's state update).

    The prior point is the newest triangulation found by prior_points in the window's history,
    or, where it finds none, stored (point, found, age): a stream's state, the joint's last
    triangulation however old. Its age input is capped at max_prior_age_s (the oldest training has shown)."""
    b, q = query.shape
    origin, direction, has, fit, count, age, oldest, uv, newest_direction, rms, residual = rays_at(events, query, config)
    point, ok = triangulate(origin, direction, has)                                         # [B,Q,2,21,3], [B,Q,2,21]
    prior, found, prior_age = prior_points(events, query, config)                       # the joint's own past
    if stored is not None:
        prior, found, prior_age = merge_priors((prior, found, prior_age), stored)
    observed = torch.where(has, age, FAR).amin(-1)                                          # newest detection used now
    # A joint that fails now: the point of the ray passing closest to its prior point, nearest to it.
    miss = torch.where(has, ray_residuals(prior, origin, direction), FAR)                # [B,Q,2,21,3]
    best = miss.argmin(-1, keepdim=True)
    pick = lambda value: value.gather(-2, best[..., None].expand(*best.shape, 3))[..., 0, :]
    o, d = pick(origin), pick(direction)
    projected = o+((prior-o)*d).sum(-1, keepdim=True).clamp_min(0.)*d
    seen = has.any(-1)
    on_ray = ~ok & found & seen
    at_prior = ~ok & found & ~seen
    anchor = torch.where(ok[..., None], point, torch.where(on_ray[..., None], projected,
                                                           torch.where(at_prior[..., None], prior, 0.)))
    kind = ok.long()+2*on_ray.long()+3*at_prior.long()
    relative = anchor[..., None, :]-origin
    miss_vector = relative-direction*(relative*direction).sum(-1, keepdim=True)
    flag = lambda value: value[..., None].float()
    camera = torch.cat((flag(has), flag(fit), count[..., None]/4, age[..., None]/TIME_UNIT_S, oldest[..., None]/TIME_UNIT_S,
                        uv*2-1, miss_vector*MISS_SCALE, (direction-newest_direction)*MISS_SCALE,
                        rms[..., None]/RESIDUAL_UNIT, residual[..., None]/RESIDUAL_UNIT), -1)
    camera = torch.where(has[..., None], camera, 0.)                                        # [B,Q,2,21,3,15]
    prior_age = torch.where(found, prior_age.clamp_max(config.max_prior_age_s), 0.)
    joint = torch.cat((anchor, torch.where(found[..., None], anchor-prior, 0.), flag(ok), flag(on_ray),
                       flag(at_prior), flag(found), prior_age[..., None]/TIME_UNIT_S), -1)  # [B,Q,2,21,11]
    features = torch.cat((joint, camera.reshape(*camera.shape[:-2], -1)), -1)
    return anchor, kind, features.reshape(b, q, HAND_JOINTS, -1), torch.where(ok, observed, 0.)


def head(dim, out, zero=False):
    """Joint tokens [...,dim] -> [...,out]; zero: starts at zero output."""
    layers = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, out))
    if zero:
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
    return layers


class Corrector(nn.Module):
    """Joint features [N,42,F], slot tokens [N,T,dim], validity [N,T] (1/0) and ages [N,T]
    (TIME_UNIT_S) -> offsets [N,42,3], each joint's expected error log(1 + mm) [N,42] (None
    without error) and each hand's logit of being inside some camera's view [N,2] (None without
    presence).

    Per-joint MLP plus a learned joint embedding; each block cross-attends to the slot tokens
    (with an age embedding, and a null key for a query without slots), then self-attends over
    the 42 joints of both hands; zero-initialised offset head; the error head (per joint) and the
    presence head (per hand, on the mean of its joints' tokens) read the final tokens detached,
    so learning them never changes the pose."""

    def __init__(self, features, dim, heads, blocks, dropout, attention_dropout, error=True, presence=True):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(features, dim), nn.GELU(), nn.Linear(dim, dim))
        self.joints = nn.Parameter(torch.randn(HAND_JOINTS, dim)*.02)
        self.age = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, dim))
        self.null = nn.Parameter(torch.zeros(1, dim))
        self.blocks = nn.ModuleList([Block(dim, heads, dropout, attention_dropout=attention_dropout)
                                     for _ in range(blocks)])
        self.output = head(dim, 3, zero=True)
        self.error = head(dim, 1) if error else None
        self.presence = head(dim, 1) if presence else None

    def forward(self, features, tokens, valid, ages):
        x = self.input(features)+self.joints
        n, _, d = tokens.shape
        keys = torch.cat((self.null.expand(n, 1, d), (tokens+self.age(ages[..., None]))*valid[..., None]), 1)
        bias = torch.cat((torch.zeros_like(valid[:, :1]), (valid-1.)*-MASK_OFF), -1)[:, None, None]
        for block in self.blocks:
            x = block(x, keys, bias)
        error = None if self.error is None else self.error(x.detach())[..., 0]
        presence = None if self.presence is None else self.presence(
            x.detach().reshape(n, NUM_HANDS, NUM_JOINTS, -1).mean(2))[..., 0]
        return self.output(x), error, presence


class CorrectorGraph(nn.Module):
    """The exported corrector: offsets, then (with the error head) each joint's expected error and
    (with the presence head) its hand's in-view logit repeated on the hand's joints, as columns
    [N,42,3 to 5] (the runtime reads one output per graph)."""

    def __init__(self, corrector):
        super().__init__()
        self.corrector = corrector

    def forward(self, features, tokens, valid, ages):
        offset, error, presence = self.corrector(features, tokens, valid, ages)
        columns = [offset]
        if error is not None:
            columns.append(error[..., None])
        if presence is not None:
            n = presence.shape[0]
            columns.append(presence[..., None].expand(n, NUM_HANDS, NUM_JOINTS).reshape(n, HAND_JOINTS, 1))
        return torch.cat(columns, -1) if len(columns) > 1 else offset


class HandLiteV3(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or LiteV3Config()
        c = self.config
        self.encoder = EventEncoder(c.dim, True, DIRECT_JOINT_FEATURES)
        self.corrector = Corrector(LITE3_FEATURES, c.dim, c.heads, c.blocks, c.dropout, c.attention_dropout,
                                   c.error_estimate, c.presence)

    @property
    def slots(self):
        return NUM_CAMERAS*self.config.slots_per_camera

    def encode_events(self, events, targets=None):
        """Per-event finger tokens (DirectEncoded [B,T,10,dim]) of the target events (default:
        all), each from its own frame only."""
        b, e = events['camera'].shape
        if targets is None:
            targets = torch.arange(e, device=events['camera'].device)
        t = targets.numel()
        features = direct_features(events['raw'][:, targets], events['valid'][:, targets])
        camera = camera_one_hot(events['camera'][:, targets])
        tokens = self.encoder(features.reshape(b*t, NUM_HANDS, -1), camera.reshape(b*t, NUM_CAMERAS))
        return DirectEncoded(tokens.reshape(b, t, *tokens.shape[1:]))

    def slot_tokens(self, events, encoded, query):
        """Slot tokens [N,10*3K,dim], validity [N,10*3K] (1/0), ages [N,10*3K] (TIME_UNIT_S)
        and whether each query had any slot [B,Q]."""
        b, q = query.shape
        n, s = b*q, self.slots
        slot, valid = select_slots(events, query, self.config.slots_per_camera, self.config.event_span_s)
        take = lambda value: _gather(value, slot).reshape(n, s, *value.shape[2:])
        valid = valid.reshape(n, s)
        token_valid = repeat_each(valid, EVENT_TOKENS)
        tokens = torch.where(token_valid[..., None], take(encoded.tokens).reshape(n, EVENT_TOKENS*s, -1), 0.)
        ages = torch.where(valid, ((query.reshape(n, 1)-take(events['capture']))/TIME_UNIT_S).clamp_min(0.), 0.)
        return tokens, token_valid.float(), repeat_each(ages, EVENT_TOKENS), valid.any(-1).reshape(b, q)

    def estimate(self, events, encoded, query, stored=None):
        """Pose [B,Q,2,21,3], whether a query had any usable frame or slot [B,Q], each joint's
        expected error log(1 + mm) [B,Q,2,21] (None without config.error_estimate), each hand's
        in-view logit [B,Q,2] (None without config.presence), how each joint was anchored
        (ANCHOR_KINDS index) [B,Q,2,21], its anchor [B,Q,2,21,3] and, where triangulated,
        the age of the newest detection used (s) [B,Q,2,21]. stored: a stream's state (see
        anchor_features). The geometry runs in float32 even under autocast."""
        b, q = query.shape
        c = self.config
        with torch.no_grad(), float32(query.device):                                        # no learned parameters
            anchor, kind, features, observed = anchor_features(events, query.float(), c, stored)
        usable = (events['present'][:, None] & (events['arrival'][:, None] <= query[..., None]+ARRIVAL_TOLERANCE_S)
                  & (events['capture'][:, None] >= query[..., None]-c.fit_span_s)).any(-1)
        tokens, token_valid, ages, any_slot = self.slot_tokens(events, encoded, query)
        offset, error, presence = self.corrector(features.reshape(b*q, HAND_JOINTS, -1), tokens, token_valid, ages)
        pose = anchor+offset.float().reshape(b, q, NUM_HANDS, NUM_JOINTS, 3)
        if error is not None:
            error = error.float().reshape(b, q, NUM_HANDS, NUM_JOINTS)
        if presence is not None:
            presence = presence.float().reshape(b, q, NUM_HANDS)
        return pose, usable | any_slot, error, presence, kind, anchor, observed

    def decode_queries(self, events, encoded, query):
        """Without a state: pose [B,Q,2,21,3], None (no camera calibration), accepted [B,Q], None
        (one stage), expected error, anchor kind, anchor and in-view logit (see estimate)."""
        pose, accepted, error, presence, kind, anchor, _ = self.estimate(events, encoded, query)
        return pose, None, accepted, None, error, kind, anchor, presence

    def forward(self, event_features, event_valid, event_camera, event_capture, event_arrival, event_present,
                query_times, return_details=False):
        """Model inputs (contracts.MODEL_INPUT_KEYS). return_details gives ModelOutput."""
        if event_features.ndim != 5 or event_features.shape[2:] != (NUM_HANDS, NUM_JOINTS, FEATURE_CHANNELS):
            raise ValueError('Expected event features [B,E,2,21,14]')
        if event_valid.shape != event_features.shape[:-1] or query_times.ndim != 2:
            raise ValueError('Invalid event/query dimensions')
        events = make_events(event_features, event_valid, event_camera, event_capture, event_arrival, event_present)
        output = ModelOutput(*self.decode_queries(events, self.encode_events(events), query_times.float()))
        return output if return_details else output.pose


class LiteV3Stream(EventStream):
    """HandLiteV3 event by event: each event's tokens are computed once on arrival; a query runs
    the geometry and the corrector.

    State: each joint's last triangulation (point, and the absolute capture time of the newest
    detection it used), updated whenever the joint triangulates. A query uses it as the prior
    point only when the history search finds none, so a joint seen by one camera keeps its depth
    however long; wherever the search finds one, the inputs (and pose) equal forward()'s, as in
    training. Nothing learned is fed back."""

    def reset(self):
        super().reset()
        shape = (NUM_HANDS, NUM_JOINTS)
        self.prior_point = torch.zeros(*shape, 3, device=self.device)
        self.prior_capture = torch.zeros(shape, dtype=torch.float64, device=self.device)
        self.prior_known = torch.zeros(shape, dtype=torch.bool, device=self.device)

    def decode(self, time):
        encoded = self.store.encoded.map(lambda value: value[None])
        query = torch.zeros(1, 1, device=self.device)
        stored = (self.prior_point[None, None], self.prior_known[None, None],
                  (time-self.prior_capture).float()[None, None])
        pose, _, _, _, kind, anchor, observed = self.model.estimate(relative_events(self.store.raw, time), encoded, query, stored)
        triangulated = kind[0, 0] == 1
        self.prior_point = torch.where(triangulated[..., None], anchor[0, 0], self.prior_point)
        self.prior_capture = torch.where(triangulated, time-observed[0, 0].double(), self.prior_capture)
        self.prior_known = self.prior_known | triangulated
        return pose[0, 0]
