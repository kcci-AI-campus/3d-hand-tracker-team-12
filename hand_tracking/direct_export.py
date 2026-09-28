"""Export HandDirect's two fixed-shape networks for direct_runtime: ONNX and ncnn.

    encoder  features [2, 21*10], camera one-hot [3]                     -> tokens [10, dim]
    query    tokens [10*3K, dim], token validity 0/1 [10*3K], ages [10*3K] -> joints [42, 3]
"""
from dataclasses import asdict
from pathlib import Path
import torch
from .constants import NUM_CAMERAS, NUM_HANDS, NUM_JOINTS
from .direct import DIRECT_JOINT_FEATURES
from .direct_runtime import EXPORT_FORMAT, GRAPHS, META_FILE
from .lite_export import FORMATS, write_graphs


def graphs(model):
    """name -> (module, example inputs with a batch of one)."""
    model = model.eval()
    tokens = model.config.event_tokens*model.slots
    torch.manual_seed(0)
    return dict(encoder=(model.encoder, (torch.randn(1, NUM_HANDS, NUM_JOINTS*DIRECT_JOINT_FEATURES),
                                         torch.eye(NUM_CAMERAS)[:1])),
                query=(model.query_network, (torch.randn(1, tokens, model.config.dim), torch.ones(1, tokens),
                                             torch.rand(1, tokens))))


def export_direct(model, directory, formats=FORMATS):
    """Write the graphs and direct.json (config) to a new directory; returns the metadata."""
    unknown = set(formats)-set(FORMATS)
    if unknown:
        raise ValueError(f'Unknown formats {sorted(unknown)}')
    directory = Path(directory)
    directory.mkdir(parents=True)
    return write_graphs(graphs(model), directory, formats, GRAPHS, META_FILE,
                        dict(architecture='direct', export_format=EXPORT_FORMAT, config=asdict(model.config)))
