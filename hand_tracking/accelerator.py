"""Devices: CPU/CUDA as plain PyTorch, and TPU through PyTorch/XLA (device type 'xla').

XLA traces each step into a graph and compiles it once per distinct tensor shape, so on
XLA: batches keep their fixed padded shapes (no trim_padding, train drops the last partial
batch), host reads (.item(), bool(tensor)) wait for the device and are left to log points,
and each step ends with sync(). torch_xla is imported only when an XLA device is used.
"""
import torch


def is_xla(device):
    return torch.device(device).type == 'xla'


def xla_device(precision='highest'):
    """The TPU device. precision: matmul precision of float32 ('highest' keeps full float32;
    TPUs otherwise may multiply in bfloat16 passes, too coarse for mm-level joints)."""
    import torch_xla
    backends = getattr(torch_xla, 'backends', None)
    if precision and hasattr(backends, 'set_mat_mul_precision'):
        backends.set_mat_mul_precision(precision)
    if hasattr(torch_xla, 'device'):
        return torch_xla.device()
    import torch_xla.core.xla_model as xm
    return xm.xla_device()


def sync(device):
    """End an XLA step: compile (once per shape) and run the traced graph. No-op elsewhere."""
    if not is_xla(device):
        return
    import torch_xla
    if hasattr(torch_xla, 'sync'):
        torch_xla.sync()
    else:
        import torch_xla.core.xla_model as xm
        xm.mark_step()


def device_batches(loader, device):
    """Batches on device: on XLA, a background thread uploads the next batches while the
    TPU runs (MpDeviceLoader, which also ends a step per batch); elsewhere the loader."""
    if not is_xla(device):
        return loader
    from torch_xla.distributed.parallel_loader import MpDeviceLoader
    return MpDeviceLoader(loader, device)


def manual_seed(device, seed):
    if is_xla(device):
        import torch_xla.core.xla_model as xm
        xm.set_rng_state(seed)


def rng_state(device):
    """The XLA random generator's state (dropout), None elsewhere."""
    if not is_xla(device):
        return None
    import torch_xla.core.xla_model as xm
    return xm.get_rng_state()


def to_cpu(value):
    """Tensors (in nested dicts/lists/tuples) moved to CPU, so checkpoints load without torch_xla."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_cpu(item) for item in value)
    return value
