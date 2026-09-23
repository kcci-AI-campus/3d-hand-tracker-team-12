"""Model and input-layout settings for both architectures. Torch-free, so deployment
runtimes can read them. The CLI is generated from a config's fields, so a new field is
automatically a training flag (see add_model_arguments)."""
from dataclasses import dataclass, field, fields
from typing import Optional


# Arrival can trail capture by the pipeline delay; windows carry this much extra history.
ARRIVAL_MARGIN_S = .25


def _option(default, help):
    return field(default=default, metadata=dict(help=help))


@dataclass
class ModelConfig:
    dim: int = _option(96, 'Token width')
    heads: int = _option(4, 'Attention heads')
    blocks: int = _option(2, 'Decoder blocks (cross-attention to events, then over joints)')
    encoder_layers: int = _option(1, 'Per-event spatial layers over the 42 joints')
    dropout: float = _option(.1, 'Dropout probability')
    # Scale of the learned per-camera correction (expected rig tolerance) and of the
    # calibration auxiliary loss; defaults match the simulator's error model.
    calibration_rotation_deg: float = _option(5., 'Learned calibration rotation scale (degrees)')
    calibration_position: float = _option(.1, 'Learned calibration shift scale (world units)')
    # Learned calibration at each event from the evidence of events that arrived within
    # calibration_span_s before it (no long-term memory).
    calibration_head: bool = _option(True, 'Learn a per-event camera calibration')
    calibration_span_s: float = _option(.5, 'Evidence window of the learned calibration (s)')
    # A query attends to events captured at most event_span_s before it.
    event_span_s: float = _option(.4, 'Events a query attends to (capture age, s)')
    # A triangulation sample combines each camera's latest ray captured within this age.
    sample_max_age_s: float = _option(.2, 'Oldest ray combined into a triangulation sample (s)')
    # Line fit over samples this recent, extrapolated to the query (0 holds the last sample);
    # samples older than anchor_hold_s are not used.
    anchor_lookback_s: float = _option(.2, 'Anchor velocity fit window (s); 0 holds the last sample')
    anchor_hold_s: float = _option(.3, 'Oldest sample an anchor may hold (s)')
    # Optional (off by default: both lowered anchor accuracy on the simulator data):
    # reject samples with an angular ray residual above this (radians), and fit
    # constant-velocity motion directly to individual rays.
    anchor_ray_gate: Optional[float] = _option(None, 'Reject samples above this angular residual (rad)')
    anchor_motion_fit: bool = _option(False, 'Fit constant-velocity motion to individual rays')

    def __post_init__(self):
        if (self.dim < 1 or self.heads < 1 or self.dim % self.heads or self.blocks < 1
                or self.encoder_layers < 0 or not 0 <= self.dropout < 1
                or min(self.anchor_lookback_s, self.calibration_span_s) < 0
                or min(self.event_span_s, self.sample_max_age_s, self.anchor_hold_s,
                       self.calibration_rotation_deg, self.calibration_position) <= 0
                or (self.anchor_ray_gate is not None and self.anchor_ray_gate <= 0)):
            raise ValueError('Invalid model configuration')

    @property
    def context_s(self):
        """Seconds of events a window needs before its first query so every query gets
        exactly what a stream would give it (anchors, event tokens and their calibration)."""
        calibration = self.calibration_span_s if self.calibration_head else 0.
        return (max(self.event_span_s, self.anchor_hold_s, self.anchor_lookback_s) + ARRIVAL_MARGIN_S
                + calibration + self.sample_max_age_s)


@dataclass
class LiteConfig:
    """HandLite: fixed-size, export-friendly model (hand_tracking.lite)."""
    dim: int = _option(64, 'Token width')
    heads: int = _option(4, 'Attention heads')
    blocks: int = _option(2, 'Decoder blocks (cross-attention to hand tokens, then over joints)')
    dropout: float = _option(.1, 'Dropout probability')
    # A query uses each camera's slots_per_camera most recent events captured within
    # event_span_s: a fixed-size input whatever the frame rate or number of events.
    slots_per_camera: int = _option(8, 'Most recent events per camera a query uses')
    event_span_s: float = _option(.5, 'Oldest event capture a query uses (s)')
    calibration_head: bool = _option(True, 'Learn a per-query camera calibration from the slots')
    calibration_rotation_deg: float = _option(5., 'Learned calibration rotation scale (degrees)')
    calibration_position: float = _option(.1, 'Learned calibration shift scale (world units)')
    sample_max_age_s: float = _option(.2, 'Oldest ray combined into a triangulation sample (s)')
    anchor_lookback_s: float = _option(.2, 'Anchor velocity fit window (s); 0 holds the last sample')
    anchor_hold_s: float = _option(.3, 'Oldest sample an anchor may hold (s)')

    def __post_init__(self):
        if (self.dim < 1 or self.heads < 1 or self.dim % self.heads or self.blocks < 1 or not 0 <= self.dropout < 1
                or self.slots_per_camera < 1 or self.anchor_lookback_s < 0
                or min(self.event_span_s, self.sample_max_age_s, self.anchor_hold_s,
                       self.calibration_rotation_deg, self.calibration_position) <= 0):
            raise ValueError('Invalid model configuration')

    @property
    def context_s(self):
        """Seconds of events a window needs before its first query (slots and their rays)."""
        return max(self.event_span_s, self.anchor_hold_s, self.anchor_lookback_s) + ARRIVAL_MARGIN_S + self.sample_max_age_s


ARCHITECTURES = {'transformer': ModelConfig, 'lite': LiteConfig}


def config_class(architecture):
    try:
        return ARCHITECTURES[architecture]
    except KeyError:
        raise ValueError(f'Unknown architecture {architecture!r}; choose from {sorted(ARCHITECTURES)}') from None


def _flag(name, default):
    """calibration_head (default True) -> --no-calibration-head; others -> --name-with-dashes."""
    if default is True:
        return '--no-'+name.replace('_', '-'), 'no_'+name
    return '--'+name.replace('_', '-'), name


def add_model_arguments(parser, config_type=ModelConfig):
    """One flag per config field, with the field's default and help."""
    group = parser.add_argument_group('model')
    for item in fields(config_type):
        flag, dest = _flag(item.name, item.default)
        if isinstance(item.default, bool):
            text = item.metadata['help']
            group.add_argument(flag, dest=dest, action='store_true', help=f'Disable: {text}' if item.default else text)
        else:
            kind = float if item.default is None else type(item.default)
            group.add_argument(flag, dest=dest, type=kind, default=item.default, help=item.metadata['help'])


def model_config_from_args(args, config_type=ModelConfig):
    values = {}
    for item in fields(config_type):
        _, dest = _flag(item.name, item.default)
        value = getattr(args, dest)
        values[item.name] = (not value) if item.default is True else value
    return config_type(**values)


@dataclass
class SamplingConfig:
    """Input layout settings stored alongside, rather than inside, model weights."""
    max_events: int = 128

    def __post_init__(self):
        if self.max_events < 1:
            raise ValueError('max_events must be positive')

    @classmethod
    def from_checkpoint(cls, saved):
        if 'sampling_config' in saved:
            return cls(**saved['sampling_config'])
        # Checkpoints saved before the module refactor stored this in CLI options.
        options = saved.get('training_options', {})
        return cls(max_events=options.get('max_events', cls.max_events))
