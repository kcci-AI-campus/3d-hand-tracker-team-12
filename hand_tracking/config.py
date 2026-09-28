"""Model and input-layout settings. Torch-free, so the deployment runtime can read them. The CLI is
generated from the config's fields, so a new field is automatically a training flag (see add_model_arguments)."""
from dataclasses import dataclass, field, fields
from .constants import NUM_HANDS, FINGERS


# Arrival can trail capture by the pipeline delay; windows carry this much extra history.
ARRIVAL_MARGIN_S = .25


def _option(default, help):
    return field(default=default, metadata=dict(help=help))


@dataclass
class LiteV3Config:
    """HandLiteV3: per camera and joint a line fitted over time to the recent detections, read at the
    query and triangulated (a failed joint anchored from its own last triangulation); a corrector network predicts each joint's offset from that anchor, and its own
    expected error, from the joints' geometry and the recent events' finger tokens
    (hand_tracking.model). No state between queries."""
    dim: int = _option(64, 'Token width of the corrector and event encoder')
    heads: int = _option(4, 'Attention heads')
    blocks: int = _option(2, 'Corrector blocks (cross-attention to event tokens, then over the 42 joints)')
    dropout: float = _option(.1, 'Dropout probability (residual branches)')
    attention_dropout: float = _option(0., 'Dropout probability of attention weights')
    fit_span_s: float = _option(.2, 'Oldest detection the line fit of each camera\'s ray uses (s)')
    # A joint that fails to triangulate now is anchored from its own newest triangulation before the
    # query: the same fit and triangulation at query - k*prior_step_s (k = 1.., within prior_span_s).
    # A stream (and the deployment runtime) also keeps each joint's last triangulation however old
    # (the state) and uses it only when the history search finds none, so its inputs equal training's
    # whenever training could have found one; its age input is capped at max_prior_age_s, the oldest
    # prior point training shows, so a long single-camera stretch stays within the trained range.
    prior_span_s: float = _option(1., 'Oldest reference time searched for a failed joint\'s last triangulation (s)')
    prior_step_s: float = _option(.2, 'Spacing of those reference times (s; fit_span_s: fit windows that do not overlap)')
    # The corrector's joints attend to each camera's slots_per_camera latest events captured
    # within event_span_s, encoded once per event (finger tokens).
    slots_per_camera: int = _option(8, 'Most recent events per camera the corrector attends to')
    event_span_s: float = _option(.5, 'Oldest event capture the corrector attends to (s)')
    # Each joint's own expected error (a confidence the pose output lacks), from its final corrector
    # token with the gradient stopped, so learning it never changes the pose.
    error_estimate: bool = _option(True, 'Also predict each joint\'s own error, log(1 + mm)')
    # Each hand's logit of being inside some camera's view (a hand always exists but can leave all three
    # cameras), from its joints' final corrector tokens (detached, so learning it never changes the pose).
    presence: bool = _option(True, 'Also predict whether each hand is inside some camera\'s view')

    def __post_init__(self):
        if (self.dim < 1 or self.heads < 1 or self.dim % self.heads or self.blocks < 0
                or not 0 <= self.dropout < 1 or not 0 <= self.attention_dropout < 1
                or self.fit_span_s <= 0 or self.slots_per_camera < 1 or self.event_span_s <= 0
                or self.prior_span_s < 0 or self.prior_step_s <= 0):
            raise ValueError('Invalid model configuration')

    @property
    def prior_steps(self):
        """Reference times searched for a failed joint's last triangulation (0: none)."""
        return int(round(self.prior_span_s/self.prior_step_s))

    @property
    def max_prior_age_s(self):
        """Oldest prior point training can show (history search's last reference plus its fit window)."""
        return self.prior_steps*self.prior_step_s+self.fit_span_s

    @property
    def context_s(self):
        """Seconds of events a window needs before its first query (the prior search's fits too)."""
        return max(self.fit_span_s+self.prior_steps*self.prior_step_s, self.event_span_s)+ARRIVAL_MARGIN_S

    @property
    def event_tokens(self):
        """Tokens the event encoder makes per event: one per finger of each hand."""
        return NUM_HANDS*FINGERS


@dataclass
class DirectConfig:
    """HandDirect: networks only, joints straight from the events (hand_tracking.direct). Kept as a
    comparison for HandLiteV3: no triangulation, anchor, error or in-view head."""
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
    # Decoder output: wrists absolute and the other joints relative to their wrist (separate
    # heads), and/or a pose after every decoder block that the next block only corrects.
    wrist_relative: bool = _option(True, 'Output each wrist absolute and the other joints relative to it')
    refine: bool = _option(True, 'Coarse to fine: every decoder block outputs a pose, the next predicts its correction')

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


# Checkpoint/export architecture names; HandLiteV3 is the default, HandDirect a networks-only comparison.
ARCHITECTURES = {'litev3': LiteV3Config, 'direct': DirectConfig}
DEFAULT_ARCHITECTURE = 'litev3'
# Fields added after checkpoints/exports were written: their value for a saved config without
# the field (the model it was built as), whatever the current default.
LEGACY_OPTIONS = {'direct': dict(wrist_relative=False, refine=False)}


def architecture_of(config):
    """The ARCHITECTURES name of a config."""
    return next(name for name, cls in ARCHITECTURES.items() if isinstance(config, cls))


def saved_config(values, architecture=DEFAULT_ARCHITECTURE):
    """Config of a saved checkpoint or export, with LEGACY_OPTIONS for fields it predates."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f'Unknown architecture {architecture!r}; choose from {sorted(ARCHITECTURES)}')
    return ARCHITECTURES[architecture](**{**LEGACY_OPTIONS.get(architecture, {}), **values})


def _flag(name, default):
    """error_estimate (default True) -> --no-error-estimate; others -> --name-with-dashes."""
    if default is True:
        return '--no-'+name.replace('_', '-'), 'no_'+name
    return '--'+name.replace('_', '-'), name


def add_model_arguments(parser, config_class=LiteV3Config):
    """One flag per config field, with the field's default and help."""
    group = parser.add_argument_group('model')
    for item in fields(config_class):
        flag, dest = _flag(item.name, item.default)
        if isinstance(item.default, bool):
            text = item.metadata['help']
            group.add_argument(flag, dest=dest, action='store_true', help=f'Disable: {text}' if item.default else text)
        else:
            group.add_argument(flag, dest=dest, type=type(item.default), default=item.default, help=item.metadata['help'])


def model_config_from_args(args, config_class=LiteV3Config):
    values = {}
    for item in fields(config_class):
        _, dest = _flag(item.name, item.default)
        value = getattr(args, dest)
        values[item.name] = (not value) if item.default is True else value
    return config_class(**values)


@dataclass
class SamplingConfig:
    """Input layout settings stored alongside, rather than inside, model weights.
    target_frame: coordinates of the pose targets, 'rig' (the nominal rig as actually placed;
    rig-wide errors no camera reveals removed, see data.rig_frame) or 'world'. mask_out_of_view: a hand
    outside all three cameras (data.hands_in_view) has no position targets; the model learns it is out of view."""
    max_events: int = 64
    target_frame: str = 'rig'
    mask_out_of_view: bool = True

    def __post_init__(self):
        if self.max_events < 1:
            raise ValueError('max_events must be positive')
        if self.target_frame not in ('rig', 'world'):
            raise ValueError("target_frame must be 'rig' or 'world'")
