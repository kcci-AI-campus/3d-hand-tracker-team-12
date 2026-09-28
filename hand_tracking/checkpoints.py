"""Checkpoints: model and input layout together (including pre-refactor checkpoints), and
the full training state for resuming. `architecture` names the model class."""
from dataclasses import asdict, dataclass
import torch
from .config import SamplingConfig, saved_config
from .direct import DirectStream, HandDirect
from .lite import HandLite, LiteStream

MODELS = {'lite': HandLite, 'direct': HandDirect}
STREAMS = {HandLite: LiteStream, HandDirect: DirectStream}


def architecture_of(model):
    return next(name for name, model_type in MODELS.items() if isinstance(model, model_type))


def build_model(architecture, config):
    return MODELS[architecture](config)


def stream_for(model):
    """Event-by-event runner of the model (results equal forward())."""
    return STREAMS[type(model)](model)


INPUT_CONTRACT = ('events [B,E,2,21,14]+valid+camera+capture/arrival seconds+present, query '
                  'times [B,Q] seconds -> pose [B,Q,2,21,3], calibration per query [B,Q,3,6] (lite); '
                  'or stream_for(model) event by event; XYZ world units')


@dataclass
class LoadedCheckpoint:
    model: torch.nn.Module
    sampling: SamplingConfig
    saved: dict


def save_checkpoint(path, value):
    """Write atomically: a crash never leaves a truncated checkpoint at path."""
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def load_checkpoint(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    architecture = saved.get('architecture')
    if architecture not in MODELS:
        raise ValueError(f'Unsupported checkpoint architecture {architecture!r} '
                         f'(HandTransformer was removed); choose from {sorted(MODELS)}')
    model = build_model(architecture, saved_config(architecture, saved['model_config'])).to(device)
    model.load_state_dict(saved['model'])
    model.eval()
    return LoadedCheckpoint(model, SamplingConfig.from_checkpoint(saved), saved)


def training_checkpoint(model, sampling, optimizer, scheduler, scaler, epoch, best_val_mpjpe_mm, options):
    """Everything needed to resume after `epoch` (0-based) and to load the model alone."""
    cuda = next(model.parameters()).device.type == 'cuda'
    return dict(model=model.state_dict(), architecture=architecture_of(model), model_config=asdict(model.config),
                sampling_config=asdict(sampling),
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                epoch=epoch, best_val_mpjpe_mm=best_val_mpjpe_mm, rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if cuda else None, training_options=options,
                torch_version=str(torch.__version__), parameters=sum(p.numel() for p in model.parameters()),
                input_contract=INPUT_CONTRACT)


def restore_training(path, model, optimizer, scheduler, scaler):
    """Load a training_checkpoint into existing objects; returns (next epoch, best val MPJPE)."""
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(saved['model'])
    optimizer.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler'])
    scaler.load_state_dict(saved['scaler'])
    torch.set_rng_state(saved['rng'])
    if next(model.parameters()).device.type == 'cuda' and saved['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return saved['epoch']+1, saved['best_val_mpjpe_mm']
