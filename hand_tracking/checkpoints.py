"""Checkpoints: model and input layout together, and the full training state for resuming."""
from dataclasses import asdict, dataclass
import torch
from .accelerator import is_xla, manual_seed, rng_state, to_cpu
from .config import ARCHITECTURES, SamplingConfig, architecture_of, saved_config
from .direct import DirectStream, HandDirect
from .model import HandLiteV3, LiteV3Stream

# Architecture name -> (model, event-by-event stream); config.ARCHITECTURES holds their configs.
MODELS = {'litev3': (HandLiteV3, LiteV3Stream), 'direct': (HandDirect, DirectStream)}
assert set(MODELS) == set(ARCHITECTURES)

INPUT_CONTRACT = ('events [B,E,2,21,14]+valid+camera+capture/arrival seconds+present, query '
                  'times [B,Q] seconds -> pose [B,Q,2,21,3] and expected joint error [B,Q,2,21]; '
                  'or stream_for(model) event by event; XYZ world units')


def build_model(config):
    """The model of a config (config.ARCHITECTURES)."""
    return MODELS[architecture_of(config)][0](config)


def stream_for(model):
    """Event-by-event runner of the model (results equal forward())."""
    return MODELS[architecture_of(model.config)][1](model)


@dataclass
class LoadedCheckpoint:
    model: torch.nn.Module
    sampling: SamplingConfig
    saved: dict


def save_checkpoint(path, value):
    """Write atomically: a crash never leaves a truncated checkpoint at path. Tensors are
    saved on CPU (XLA tensors would need torch_xla to load)."""
    temporary = path.with_suffix('.tmp')
    torch.save(to_cpu(value), temporary)
    temporary.replace(path)


def load_checkpoint(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('architecture') not in MODELS:
        raise ValueError(f"Not a checkpoint of {sorted(MODELS)}: {saved.get('architecture')!r}")
    model = build_model(saved_config(saved['model_config'], saved['architecture'])).to(device)
    model.load_state_dict(saved['model'])
    model.eval()
    return LoadedCheckpoint(model, SamplingConfig(**saved['sampling_config']), saved)


def training_checkpoint(model, sampling, optimizer, scheduler, scaler, epoch, best_val_mpjpe_mm, options):
    """Everything needed to resume after `epoch` (0-based) and to load the model alone."""
    device = next(model.parameters()).device
    cuda = device.type == 'cuda'
    return dict(model=model.state_dict(), architecture=architecture_of(model.config), model_config=asdict(model.config),
                sampling_config=asdict(sampling),
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                epoch=epoch, best_val_mpjpe_mm=best_val_mpjpe_mm, rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if cuda else None, xla_rng=rng_state(device),
                training_options=options,
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
    device = next(model.parameters()).device
    if device.type == 'cuda' and saved['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    if is_xla(device) and saved.get('xla_rng') is not None:
        manual_seed(device, int(saved['xla_rng']))
    return saved['epoch']+1, saved['best_val_mpjpe_mm']
