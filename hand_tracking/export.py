"""ONNX export of HandTransformer.forward with dynamic batch, event and query counts.

The exported graph takes the seven model inputs (contracts.MODEL_INPUT_KEYS) and returns
pose [B,Q,2,21,3] and per-event calibration [B,E,3,6]. It runs the same code as forward():
the model path avoids operators without an ONNX equivalent (linalg.solve, cross, cummax,
the SDPA decomposition). Needs `pip install -r requirements-export.txt`.

Not exportable: anchor_motion_fit (Cholesky and 6x6 solves). Streaming (EventStream)
stays in PyTorch; the exported graph recomputes every event per call, as forward() does.
"""
from pathlib import Path
import torch
from torch import nn
from .constants import NUM_CAMERAS, NUM_HANDS, NUM_JOINTS, FEATURE_CHANNELS, RAY_ORIGIN, RAY_DIRECTION
from .contracts import MODEL_INPUT_KEYS

OUTPUT_NAMES = ('pose', 'calibration')


class OnnxHandTransformer(nn.Module):
    """forward() with tensor-only outputs and input names matching MODEL_INPUT_KEYS."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, event_features, event_valid, event_camera, event_capture, event_arrival, event_present,
                query_times):
        output = self.model(event_features, event_valid, event_camera, event_capture, event_arrival,
                            event_present, query_times, return_details=True)
        return output.pose, output.calibration


def example_inputs(batch=2, events=6, queries=2):
    """Small valid inputs for tracing. Sizes above 1 keep every axis dynamic in torch.export."""
    features = torch.zeros(batch, events, NUM_HANDS, NUM_JOINTS, FEATURE_CHANNELS)
    camera = (torch.arange(events) % NUM_CAMERAS).expand(batch, events).contiguous()
    origins = torch.tensor([[-1., -1., .5], [1., -1., .5], [0., 1., 1.]])
    features[..., RAY_ORIGIN] = origins[camera][:, :, None, None]
    features[..., RAY_DIRECTION] = nn.functional.normalize(-origins[camera], dim=-1)[:, :, None, None]
    capture = (torch.arange(events)*.02-.12).expand(batch, events).contiguous()
    return (features, torch.ones(batch, events, NUM_HANDS, NUM_JOINTS, dtype=torch.bool), camera, capture,
            capture+.03, torch.ones(batch, events, dtype=torch.bool),
            torch.linspace(-.02, 0., queries).expand(batch, queries).contiguous())


def export_onnx(model, path, opset=None):
    """Write model (eval mode) to path as ONNX; returns the path."""
    if model.config.anchor_motion_fit:
        raise ValueError('anchor_motion_fit uses Cholesky and 6x6 solves, which have no ONNX operators')
    model = model.eval()
    batch, events, queries = torch.export.Dim('batch'), torch.export.Dim('events'), torch.export.Dim('queries')
    per_event = {0: batch, 1: events}
    dynamic = dict(zip(MODEL_INPUT_KEYS, [per_event]*6+[{0: batch, 1: queries}]))
    path = Path(path)
    options = {} if opset is None else dict(opset_version=opset)
    with torch.no_grad():
        program = torch.onnx.export(OnnxHandTransformer(model), example_inputs(), dynamo=True,
                                    dynamic_shapes=dynamic, output_names=list(OUTPUT_NAMES), verbose=False,
                                    **options)
    program.save(str(path))
    return path


def compare_onnx(path, model, inputs):
    """Largest |ONNX Runtime - PyTorch| over pose and calibration for these inputs."""
    import onnxruntime
    session = onnxruntime.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    names = [value.name for value in session.get_inputs()]
    if tuple(names) != MODEL_INPUT_KEYS:
        raise ValueError(f'Unexpected ONNX inputs {names}')
    actual = session.run(list(OUTPUT_NAMES), {name: value.numpy() for name, value in zip(names, inputs)})
    with torch.no_grad():
        expected = OnnxHandTransformer(model.eval())(*inputs)
    return max(float(abs(a-e.numpy()).max()) for a, e in zip(actual, expected))
