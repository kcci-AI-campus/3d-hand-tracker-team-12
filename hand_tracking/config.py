"""Model and input-layout settings. Torch-free, so deployment runtimes can read them. The CLI is generated from a config's fields, so a new field is
automatically a training flag (see add_model_arguments)."""
from dataclasses import dataclass, field, fields
from .constants import NUM_HANDS, FINGERS


# Arrival can trail capture by the pipeline delay; windows carry this much extra history.
ARRIVAL_MARGIN_S = .25


def _option(default, help):
    return field(default=default, metadata=dict(help=help))


@dataclass
class LiteConfig:
    """HandLite: fixed-size, export-friendly model (hand_tracking.lite)."""
    dim: int = _option(64, 'Token width')
    heads: int = _option(4, 'Attention heads')
    blocks: int = _option(2, 'Decoder blocks (cross-attention to hand tokens, then over joints)')
    dropout: float = _option(.1, 'Dropout probability (residual branches)')
    attention_dropout: float = _option(0., 'Dropout probability of attention weights (costly on large maps)')
    # A query uses each camera's slots_per_camera most recent events captured within
    # event_span_s: a fixed-size input whatever the frame rate or number of events.
    slots_per_camera: int = _option(8, 'Most recent events per camera a query uses')
    event_span_s: float = _option(.5, 'Oldest event capture a query uses (s)')
    calibration_head: bool = _option(True, 'Learn a per-query camera calibration from the slots')
    calibration_rotation_deg: float = _option(5., 'Learned calibration rotation scale (degrees)')
    calibration_position: float = _option(.1, 'Learned calibration shift scale (world units)')
    gap_embedding: bool = _option(True, 'Add a learned capture-to-query gap embedding to decoder hand tokens')
    finger_tokens: bool = _option(True, 'Encode each hand as one token per finger (with the wrist), not one token')
    sample_max_age_s: float = _option(.2, 'Oldest ray combined into a triangulation sample (s)')
    anchor_lookback_s: float = _option(.2, 'Anchor velocity fit window (s); 0 holds the last sample')
    anchor_hold_s: float = _option(.3, 'Oldest sample an anchor may hold (s)')
    # A joint seen by fewer than two cameras is anchored on its newest ray, at the depth of
    # its anchor or of its last triangulation within anchor_depth_memory_s.
    anchor_ray_depth: bool = _option(True, 'Anchor joints seen by one camera on that ray at a remembered depth')
    anchor_depth_memory_s: float = _option(1., 'Oldest triangulation whose depth a single-ray anchor keeps (s)')
    # Query-time triangulation of calibrated rays drops one ray of three or more when it is
    # this many times further from the others' point than they are (0: off). Not applied to
    # the arrival-time nominal triangulation: a miscalibrated camera looks like an outlier.
    # Off by default: on clean simulated windows it cost 0.2 mm with the true calibration but
    # 13 mm with none (a calibrator not yet trained); 6 separated hand-swap outliers well.
    ray_outlier_ratio: float = _option(0., 'Drop a ray this many times less consistent than the others (0: off; try 6)')

    def __post_init__(self):
        if (self.dim < 1 or self.heads < 1 or self.dim % self.heads or self.blocks < 1 or not 0 <= self.dropout < 1
                or not 0 <= self.attention_dropout < 1
                or self.slots_per_camera < 1 or self.anchor_lookback_s < 0 or self.ray_outlier_ratio < 0
                or min(self.event_span_s, self.sample_max_age_s, self.anchor_hold_s,
                       self.calibration_rotation_deg, self.calibration_position, self.anchor_depth_memory_s) <= 0):
            raise ValueError('Invalid model configuration')

    @property
    def context_s(self):
        """Seconds of events a window needs before its first query (slots, remembered
        triangulations and their rays)."""
        memory = self.anchor_depth_memory_s if self.anchor_ray_depth else 0.
        return (max(self.event_span_s, self.anchor_hold_s, self.anchor_lookback_s, memory) + ARRIVAL_MARGIN_S
                + self.sample_max_age_s)

    @property
    def anchor_features(self):
        """Decoder input per joint: anchor (3), flags (5) and, with anchor_ray_depth, single-ray (1)."""
        return 8+self.anchor_ray_depth

    @property
    def event_tokens(self):
        """Tokens the encoder makes per event: per hand one per finger, or one."""
        return NUM_HANDS*(FINGERS if self.finger_tokens else 1)


@dataclass
class DirectConfig:
    """HandDirect: networks only, joints straight from the events (hand_tracking.direct)."""
    dim: int = _option(64, 'Token width')
    heads: int = _option(4, 'Attention heads')
    fusion_blocks: int = _option(2, 'Self-attention blocks over the slot tokens (views and times)')
    blocks: int = _option(2, 'Decoder blocks (cross-attention to the tokens, then over joints)')
    dropout: float = _option(.1, 'Dropout probability (residual branches)')
    attention_dropout: float = _option(0., 'Dropout probability of attention weights')
    # A query uses each camera's slots_per_camera most recent events captured within
    # event_span_s: a fixed-size input whatever the frame rate or number of events.
    slots_per_camera: int = _option(8, 'Most recent events per camera a query uses')
    event_span_s: float = _option(.5, 'Oldest event capture a query uses (s)')

    def __post_init__(self):
        if (self.dim < 1 or self.heads < 1 or self.dim % self.heads or self.blocks < 1 or self.fusion_blocks < 0
                or not 0 <= self.dropout < 1 or not 0 <= self.attention_dropout < 1 or self.slots_per_camera < 1
                or self.event_span_s <= 0):
            raise ValueError('Invalid model configuration')

    @property
    def context_s(self):
        """Seconds of events a window needs before its first query."""
        return self.event_span_s+ARRIVAL_MARGIN_S

    @property
    def event_tokens(self):
        """Tokens the encoder makes per event: one per finger of each hand."""
        return NUM_HANDS*FINGERS


ARCHITECTURES = {'lite': LiteConfig, 'direct': DirectConfig}
# Fields added after checkpoints/exports were written: their value for a saved config
# without the field (the model it was built as), whatever the current default.
LEGACY_OPTIONS = {'lite': dict(gap_embedding=False, finger_tokens=False, anchor_ray_depth=False, ray_outlier_ratio=0.)}


def config_class(architecture):
    try:
        return ARCHITECTURES[architecture]
    except KeyError:
        raise ValueError(f'Unknown architecture {architecture!r}; choose from {sorted(ARCHITECTURES)}') from None


def saved_config(architecture, values):
    """Config of a saved checkpoint or export, with LEGACY_OPTIONS for fields it predates."""
    return config_class(architecture)(**{**LEGACY_OPTIONS.get(architecture, {}), **values})


def _flag(name, default):
    """calibration_head (default True) -> --no-calibration-head; others -> --name-with-dashes."""
    if default is True:
        return '--no-'+name.replace('_', '-'), 'no_'+name
    return '--'+name.replace('_', '-'), name


def add_model_arguments(parser, config_type=LiteConfig):
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


def model_config_from_args(args, config_type=LiteConfig):
    values = {}
    for item in fields(config_type):
        _, dest = _flag(item.name, item.default)
        value = getattr(args, dest)
        values[item.name] = (not value) if item.default is True else value
    return config_type(**values)


@dataclass
class SamplingConfig:
    """Input layout settings stored alongside, rather than inside, model weights.
    target_frame: coordinates of the pose targets, 'rig' (the nominal rig as actually placed;
    rig-wide errors no camera reveals removed, see data.rig_frame) or 'world'."""
    max_events: int = 128
    target_frame: str = 'rig'

    def __post_init__(self):
        if self.max_events < 1:
            raise ValueError('max_events must be positive')
        if self.target_frame not in ('rig', 'world'):
            raise ValueError("target_frame must be 'rig' or 'world'")

    @classmethod
    def from_checkpoint(cls, saved):
        if 'sampling_config' in saved:
            # Checkpoints before target_frame were trained on world-frame targets.
            return cls(**{'target_frame': 'world', **saved['sampling_config']})
        # Checkpoints saved before the module refactor stored this in CLI options.
        options = saved.get('training_options', {})
        return cls(max_events=options.get('max_events', cls.max_events))
