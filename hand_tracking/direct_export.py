"""Export HandDirect's two fixed-shape networks to ncnn (converted by pnnx) for direct_runtime.

    encoder  features [2, 21*10], camera one-hot [3]                     -> tokens [10, dim]
    query    tokens [10*3K, dim], token validity 0/1 [10*3K], ages [10*3K] -> joints [42, 3]
"""
from dataclasses import asdict
import json
from pathlib import Path
import torch
from .constants import NUM_CAMERAS, NUM_HANDS, NUM_JOINTS
from .deploy import _pnnx
from .direct_runtime import EXPORT_FORMAT, GRAPHS, META_FILE
from .networks import DIRECT_JOINT_FEATURES


def graphs(model):
    """name -> (module, example inputs with a batch of one)."""
    model = model.eval()
    tokens = model.config.event_tokens*model.slots
    torch.manual_seed(0)
    return dict(encoder=(model.encoder, (torch.randn(1, NUM_HANDS, NUM_JOINTS*DIRECT_JOINT_FEATURES),
                                         torch.eye(NUM_CAMERAS)[:1])),
                query=(model.query_network, (torch.randn(1, tokens, model.config.dim), torch.ones(1, tokens),
                                             torch.rand(1, tokens))))


def export_direct(model, directory):
    """Write the ncnn graphs and direct.json (config and graph input shapes) to a new directory;
    returns the metadata."""
    directory = Path(directory)
    directory.mkdir(parents=True)
    exported = graphs(model)
    with torch.no_grad():
        for name in GRAPHS:
            _pnnx(*exported[name], directory, name)
    meta = dict(architecture='direct', export_format=EXPORT_FORMAT, config=asdict(model.config), graphs={
        name: dict(inputs=[list(value.shape[1:]) for value in exported[name][1]]) for name in GRAPHS})
    (directory/META_FILE).write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return meta
