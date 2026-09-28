"""Export HandLiteV3's networks to ncnn (converted by pnnx) for hand_tracking.runtime. Shapes have
no batch axis in ncnn. T = 10 * 3 * slots_per_camera, F = model.LITE3_FEATURES.

    encoder    event features [2, 21*10], camera one-hot [3]                   -> finger tokens [10, dim]
    corrector  joint features [42, F], slot tokens [T, dim], validity 0/1 [T], ages [T]
               -> offsets [42, 3], then with config.error_estimate each joint's expected error log(1 + mm) and
                  with config.presence its hand's in-view logit: [42, 3 to 5]
"""
from dataclasses import asdict
import contextlib
import json
import os
from pathlib import Path
import torch
from .constants import NUM_CAMERAS, NUM_HANDS, NUM_JOINTS, HAND_JOINTS
from .model import EVENT_TOKENS, LITE3_FEATURES, CorrectorGraph
from .networks import DIRECT_JOINT_FEATURES
from .runtime import EXPORT_FORMAT, GRAPHS, META_FILE


def graphs(model):
    """name -> (module, example inputs with a batch of one)."""
    model = model.eval()
    torch.manual_seed(0)
    tokens = model.slots*EVENT_TOKENS
    return dict(encoder=(model.encoder, (torch.randn(1, NUM_HANDS, NUM_JOINTS*DIRECT_JOINT_FEATURES),
                                         torch.eye(NUM_CAMERAS)[:1])),
                corrector=(CorrectorGraph(model.corrector), (torch.randn(1, HAND_JOINTS, LITE3_FEATURES),
                                                             torch.randn(1, tokens, model.config.dim),
                                                             torch.ones(1, tokens), torch.rand(1, tokens))))


def export_ncnn(model, directory):
    """Write the ncnn graphs and litev3.json (config and graph input shapes) to a new directory;
    returns the metadata."""
    directory = Path(directory)
    directory.mkdir(parents=True)
    exported = graphs(model)
    with torch.no_grad():
        for name in GRAPHS:
            _pnnx(*exported[name], directory, name)
    meta = dict(export_format=EXPORT_FORMAT, config=asdict(model.config), graphs={
        name: dict(inputs=[list(value.shape[1:]) for value in exported[name][1]]) for name in GRAPHS})
    (directory/META_FILE).write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return meta


@contextlib.contextmanager
def _working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _pnnx(module, example, directory, name):
    """Convert with pnnx and keep only the ncnn param/bin (pnnx writes next to its input)."""
    import pnnx
    with _working_directory(directory):
        pnnx.export(module, f'{name}.pt', example)
        for path in list(Path('.').glob(f'{name}*'))+list(Path('__pycache__').glob(f'{name}*')):
            if not path.name.endswith(('.ncnn.param', '.ncnn.bin')):
                path.unlink()
        cache = Path('__pycache__')
        if cache.is_dir() and not any(cache.iterdir()):
            cache.rmdir()
