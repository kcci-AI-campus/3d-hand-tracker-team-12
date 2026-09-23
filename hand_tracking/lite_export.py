"""Export HandLite's three fixed-shape networks for lite_runtime: ONNX (onnxruntime) and
ncnn (converted by pnnx). Shapes have no batch axis in ncnn; ONNX graphs take a batch of one.

    encoder     features [2, 21*14], camera one-hot [3]                  -> tokens [2, dim]
    calibrator  tokens [2*3K, dim], pool bias [3, 2*3K]                  -> correction [3, 6]
    decoder     tokens [2*3K, dim], token bias [2*3K], gaps [2*3K], anchor features [42, 8] -> offset [42, 3]
"""
from dataclasses import asdict
import contextlib
import json
import os
from pathlib import Path
import torch
from torch import nn
from .constants import NUM_CAMERAS, NUM_HANDS, HAND_JOINTS
from .lite import ANCHOR_FEATURES, HAND_FEATURES
from .lite_runtime import GRAPHS, META_FILE

FORMATS = ('onnx', 'ncnn')


class _Graph(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module


class EncoderGraph(_Graph):
    def forward(self, features, camera):
        return self.module(features, camera)


class CalibratorGraph(_Graph):
    def forward(self, tokens, pool_bias):
        return self.module(tokens, pool_bias)


class DecoderGraph(_Graph):
    def forward(self, tokens, token_bias, gaps, anchor_features):
        return self.module(tokens, token_bias, gaps, anchor_features)


def graphs(model):
    """name -> (module, example inputs with a batch of one)."""
    model = model.eval()
    dim, tokens = model.config.dim, NUM_HANDS*model.slots
    torch.manual_seed(0)
    token_example = torch.randn(1, tokens, dim)
    result = dict(encoder=(EncoderGraph(model.encoder), (torch.randn(1, NUM_HANDS, HAND_FEATURES),
                                                         torch.eye(NUM_CAMERAS)[:1])),
                  decoder=(DecoderGraph(model.decoder), (token_example, torch.zeros(1, tokens), torch.rand(1, tokens),
                                                         torch.randn(1, HAND_JOINTS, ANCHOR_FEATURES))))
    if model.calibrator is not None:
        result['calibrator'] = (CalibratorGraph(model.calibrator), (token_example, torch.zeros(1, NUM_CAMERAS, tokens)))
    return result


@contextlib.contextmanager
def _working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def export_lite(model, directory, formats=FORMATS):
    """Write the graphs and lite.json (config) to a new directory; returns the metadata."""
    unknown = set(formats)-set(FORMATS)
    if unknown:
        raise ValueError(f'Unknown formats {sorted(unknown)}')
    directory = Path(directory)
    directory.mkdir(parents=True)
    exported = graphs(model)
    if 'calibrator' not in exported:
        # The runtime always loads three graphs; without a head it never calls this one.
        exported['calibrator'] = (CalibratorGraph(_ZeroCalibration()), (torch.zeros(1, 1, model.config.dim),
                                                                          torch.zeros(1, NUM_CAMERAS, 1)))
    with torch.no_grad():
        for name in GRAPHS:
            module, example = exported[name]
            if 'onnx' in formats:
                program = torch.onnx.export(module, example, dynamo=True, verbose=False)
                program.save(str(directory/f'{name}.onnx'))
            if 'ncnn' in formats:
                _pnnx(module, example, directory, name)
    meta = dict(architecture='lite', config=asdict(model.config), formats=list(formats), graphs={
        name: dict(inputs=[list(value.shape[1:]) for value in exported[name][1]]) for name in GRAPHS})
    (directory/META_FILE).write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return meta


class _ZeroCalibration(nn.Module):
    def forward(self, tokens, pool_bias):
        return pool_bias[:, :, :1].expand(-1, -1, 6)*0.


def _pnnx(module, example, directory, name):
    """Convert with pnnx and keep only the ncnn param/bin (pnnx writes next to its input)."""
    import pnnx
    with _working_directory(directory):
        pnnx.export(module, f'{name}.pt', example)
        intermediate = [path for path in Path('.').glob(f'{name}*')
                        if not path.name.endswith(('.ncnn.param', '.ncnn.bin', '.onnx'))
                        or path.name.endswith('.pnnx.onnx')]
        intermediate += list(Path('__pycache__').glob(f'{name}*'))
        for path in intermediate:
            path.unlink()
        cache = Path('__pycache__')
        if cache.is_dir() and not any(cache.iterdir()):
            cache.rmdir()
