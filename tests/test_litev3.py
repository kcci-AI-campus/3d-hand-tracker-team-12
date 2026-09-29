"""HandLiteV3: the line-fit triangulation anchor, the corrector's outputs, batch/stream padding and
the deployment runtime (numpy geometry around the exported networks)."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from model_helpers import push_all, randomize_heads, small, synthetic_events
import torch
from hand_tracking.checkpoints import build_model, stream_for
from hand_tracking.model import ANCHOR_KINDS, CorrectorGraph, HandLiteV3, LiteV3Stream
from hand_tracking import runtime as lite_runtime

# The line fit runs in float32 (torch) or float64 (numpy runtime) and is ill-conditioned (s0*s2 - s1^2
# over a few frames), so equal inputs summed differently (padding, stream vs batch, runtime) land
# about 1e-4 to 1e-3 world units apart (under 0.4 mm at 40 cm per unit).
GEOMETRY_ATOL = 1e-3


def moving_events(seconds=1.2, rate=11.7, seed=0, missing=True):
    """Three cameras seeing both hands move linearly; some joints missing."""
    torch.manual_seed(seed)
    base, velocity = torch.randn(2,21,3)*.1, torch.randn(2,21,3)*.3
    times = np.arange(-seconds, 0, 1/rate)
    inputs = list(synthetic_events(lambda t: base+velocity*t, [t for t in times for _ in range(3)],
                                   [c for _ in times for c in range(3)]))
    if missing:
        inputs[1] = inputs[1].clone()
        inputs[1][0, ::5, 1, :7] = False
    return inputs, base, velocity


class ModelTests(unittest.TestCase):
    def test_untrained_pose_is_the_triangulated_anchor(self):
        """The offset head starts at zero, so the pose is the anchor: the line fit of exact rays of
        linear motion, triangulated, is the true position at the query."""
        model = HandLiteV3(small()).eval()
        inputs, base, velocity = moving_events(missing=False)
        query = float(inputs[4][0,-1])
        with torch.no_grad():
            output = model(*inputs, torch.tensor([[query]]), return_details=True)
        self.assertTrue((output.anchored[0,0] == ANCHOR_KINDS.index('triangulated')).all())
        torch.testing.assert_close(output.pose[0,0], output.anchor_xyz[0,0])
        # The fit is linear in the ray direction, not the position: exact to first order only.
        torch.testing.assert_close(output.pose[0,0], base+velocity*query, atol=3e-3, rtol=0)

    def test_outputs_padding_and_gradients(self):
        model = randomize_heads(HandLiteV3(small()))
        inputs, _, _ = moving_events()
        query = inputs[4][:, -3:]
        output = model.eval()(*inputs, query, return_details=True)
        self.assertEqual(output.pose.shape, (1,3,2,21,3))
        self.assertEqual(output.error.shape, (1,3,2,21))
        self.assertEqual(output.presence.shape, (1,3,2))
        self.assertTrue(output.accepted.all())
        # Padding events (any values) change nothing.
        padded = [torch.cat((value, value[:, :3]), 1) for value in inputs]
        padded[5][:, -3:] = False
        with torch.no_grad():
            torch.testing.assert_close(model(*padded, query), output.pose.detach(), atol=GEOMETRY_ATOL, rtol=0)
        # Every network learns from the pose loss except the detached error and in-view heads.
        model.train()
        out = model(*inputs, query, return_details=True)
        out.pose.square().sum().backward()
        for name in ('encoder', 'corrector.input', 'corrector.blocks', 'corrector.output'):
            grads = [p.grad for n, p in model.named_parameters() if n.startswith(name)]
            self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads), name)
        for name in ('corrector.error', 'corrector.presence'):
            self.assertTrue(all(p.grad is None for n, p in model.named_parameters() if n.startswith(name)), name)

    def test_heads_can_be_disabled(self):
        model = build_model(small(error_estimate=False, presence=False)).eval()
        inputs, _, _ = moving_events()
        with torch.no_grad():
            output = model(*inputs, inputs[4][:, -1:], return_details=True)
        self.assertIsNone(output.error)
        self.assertIsNone(output.presence)
        self.assertIsInstance(stream_for(model), LiteV3Stream)


def write_meta(folder, model):
    (Path(folder)/lite_runtime.META_FILE).write_text(json.dumps(dict(
        export_format=lite_runtime.EXPORT_FORMAT, config=asdict(model.config))))


class TorchGraphs:
    """The exported graphs' PyTorch modules, run like ncnn (inputs without a batch axis)."""

    def __init__(self, model):
        self.modules = dict(encoder=model.encoder, corrector=CorrectorGraph(model.corrector).eval())

    @torch.no_grad()
    def run(self, name, *arrays):
        return self.modules[name](*[torch.as_tensor(np.asarray(a, np.float32))[None] for a in arrays])[0].numpy()


def runtime_worst(model, runtime, inputs):
    """Largest |runtime - stream| (both with their joint state) over queries after every third
    event, then while only camera 0 still sees the hands for longer than the history search."""
    stream, worst = LiteV3Stream(model), 0.
    def compare(time):
        nonlocal worst
        with torch.inference_mode():
            expected = stream.query(time).numpy()
        worst = max(worst, float(np.abs(runtime.query(time)-expected).max()))
    for i in range(inputs[0].shape[1]):
        event = (int(inputs[2][0,i]), inputs[0][0,i].numpy(), inputs[1][0,i].numpy(),
                 float(inputs[3][0,i]), float(inputs[4][0,i]))
        assert runtime.push(*event) == stream.push(*event)
        if i % 3 == 2:
            compare(event[4])
    for step in range(1, 25):                                                 # camera 0 alone for 2 s
        time = event[4]+step*.085
        for runner in (runtime, stream):
            runner.push(0, event[1], event[2], time-.06, time)
        compare(time)
    return worst


class RuntimeTests(unittest.TestCase):
    def test_numpy_runtime_matches_stream(self):
        """The runtime's float64 numpy geometry and state equal the PyTorch stream's."""
        model = randomize_heads(HandLiteV3(small()).eval(), std=.05)
        inputs, _, _ = moving_events(seed=3)
        with tempfile.TemporaryDirectory() as folder:
            write_meta(folder, model)
            runtime = lite_runtime.LiteV3Runtime(folder, graphs=TorchGraphs(model))
            self.assertLess(runtime_worst(model, runtime, inputs), GEOMETRY_ATOL)
        self.assertEqual(runtime.error_mm.shape, (2,21))
        self.assertEqual(runtime.in_view_probability.shape, (2,))
        self.assertTrue((runtime.anchor_kind == ANCHOR_KINDS.index('ray')).any())   # single camera: on its ray

    @unittest.skipIf(any(importlib.util.find_spec(name) is None for name in ('ncnn', 'pnnx')),
                     'Install requirements-export.txt for ncnn tests')
    def test_exported_ncnn_runtime_matches_stream(self):
        from hand_tracking.deploy import export_ncnn
        model = randomize_heads(HandLiteV3(small()).eval(), std=.05)
        inputs, _, _ = moving_events(seed=3)
        with tempfile.TemporaryDirectory() as folder:
            export_ncnn(model, Path(folder)/'export')
            runtime = lite_runtime.LiteV3Runtime(Path(folder)/'export')
            self.assertLess(runtime_worst(model, runtime, inputs), 2e-3)            # ncnn may use fp16

    def test_ncnn_float64_inputs_stay_alive_until_copied(self):
        """ncnn.Mat only borrows its array's memory: the float32 copy of a float64 input must
        still exist when clone() copies it (a freed temporary gave the networks garbage)."""
        import weakref

        class Mat:
            def __init__(self, array):
                self.array = weakref.ref(array)          # borrowed, like ncnn's Mat

            def clone(self):
                array = self.array()
                if array is None:
                    raise AssertionError('Mat cloned after its array was freed')
                return array.copy()

        class Extractor:
            def input(self, name, mat):
                self.value = mat

            def extract(self, name):
                return 0, self.value

        graphs = object.__new__(lite_runtime.NcnnGraphs)
        graphs.ncnn = SimpleNamespace(Mat=Mat)
        graphs.nets = {'net': SimpleNamespace(create_extractor=Extractor)}
        value = np.linspace(0, 1, 42*56).reshape(42, 56)                    # float64, as the runtime passes
        np.testing.assert_array_equal(graphs.run('net', value), value.astype(np.float32))


if __name__ == '__main__':
    unittest.main()
