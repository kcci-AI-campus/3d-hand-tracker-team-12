"""HandDirect: networks only; batch/stream equivalence and the deployment runtime."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from model_helpers import synthetic_events
import torch
from hand_tracking.checkpoints import build_model, stream_for
from hand_tracking.config import DirectConfig
from hand_tracking.direct import DirectStream, HandDirect
from hand_tracking.events import make_events
from hand_tracking import direct_runtime

def direct(**overrides):
    return DirectConfig(**{**dict(dim=32, heads=4, blocks=1, fusion_blocks=1, dropout=0., slots_per_camera=4),
                           **overrides})


def moving_events(seconds=1.2, rate=17.1, seed=0):
    torch.manual_seed(seed)
    base, velocity = torch.randn(2,21,3)*.1, torch.randn(2,21,3)*.3
    times = np.arange(-seconds, 0, 1/rate)
    inputs = list(synthetic_events(lambda t: base+velocity*t, [t for t in times for _ in range(3)],
                                   [c for _ in times for c in range(3)]))
    inputs[1] = inputs[1].clone()
    inputs[1][0, ::5, 1, :7] = False                                  # some missing joints
    return inputs


def with_outputs(model, std=.1):
    """The output layer starts small; give it weight so outputs depend on every input."""
    torch.nn.init.normal_(model.query_network.decoder.output[-1].weight, std=std)
    return model.eval()


class DirectModelTests(unittest.TestCase):
    def test_shapes_and_each_event_encoded_alone(self):
        model = with_outputs(HandDirect(direct()))
        inputs = moving_events()
        query = inputs[4][:, -1:]+torch.tensor([[0., .03]])
        with torch.no_grad():
            output = model(*inputs, query, return_details=True)
        self.assertEqual(tuple(output.pose.shape), (1, 2, 2, 21, 3))
        self.assertIsNone(output.calibration)
        self.assertTrue(bool(output.accepted.all()))
        # No geometry across cameras: an event's tokens do not depend on any other event.
        events = make_events(*inputs)
        alone = make_events(*[value[:, -1:] for value in inputs])
        with torch.no_grad():
            torch.testing.assert_close(model.encode_events(events).tokens[:, -1], model.encode_events(alone).tokens[:, 0])

    def test_padding_changes_nothing_and_gradients_reach_every_network(self):
        model = with_outputs(HandDirect(direct()))
        inputs = moving_events()
        query = inputs[4][:, -1:]
        padded = [torch.cat((value, value[:, :5]), 1) for value in inputs]
        padded[5][:, -5:] = False
        with torch.no_grad():
            torch.testing.assert_close(model(*padded, query), model(*inputs, query), atol=1e-6, rtol=0)
        model.train()
        model(*inputs, query).square().sum().backward()
        for name, module in (('encoder', model.encoder), ('fusion', model.query_network.fusion),
                             ('decoder', model.query_network.decoder)):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads), name)
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0., name)

    def test_stream_matches_forward(self):
        model = with_outputs(HandDirect(direct()))
        self.assertIsInstance(stream_for(model), DirectStream)
        self.assertIsInstance(build_model(direct()), HandDirect)
        inputs = moving_events(seconds=2.)
        stream, epoch = DirectStream(model), 1_700_000_000.
        for i in range(inputs[0].shape[1]):
            stream.push(int(inputs[2][0,i]), inputs[0][0,i], inputs[1][0,i],
                        epoch+float(inputs[3][0,i]), epoch+float(inputs[4][0,i]))
            if i % 5 == 4:
                time = float(inputs[4][0,i])+.013
                with torch.no_grad():
                    expected = model(*[value[:, :i+1] for value in inputs], torch.tensor([[time]]))[0,0]
                torch.testing.assert_close(stream.query(epoch+time), expected, atol=1e-5, rtol=1e-5)
        self.assertLess(len(stream), inputs[0].shape[1])                    # old events pruned
        self.assertFalse(stream.push(0, inputs[0][0,0], inputs[1][0,0], epoch-5., epoch+1.))   # stale capture


class TorchGraphs:
    """The exported graphs' PyTorch modules, run like ncnn (inputs without a batch axis)."""

    def __init__(self, model):
        self.modules = dict(encoder=model.encoder, query=model.query_network)

    @torch.no_grad()
    def run(self, name, *arrays):
        return self.modules[name](*[torch.as_tensor(np.asarray(a, np.float32))[None] for a in arrays])[0].numpy()


def runtime_worst(model, runtime, inputs):
    """Largest |runtime - stream| over queries after every fifth event and after a camera outage."""
    stream, worst = DirectStream(model), 0.
    for i in range(inputs[0].shape[1]):
        event = (int(inputs[2][0,i]), inputs[0][0,i].numpy(), inputs[1][0,i].numpy(),
                 float(inputs[3][0,i]), float(inputs[4][0,i]))
        assert runtime.push(*event) == stream.push(*event)
        if i % 5 == 4:
            worst = max(worst, float(np.abs(runtime.query(event[4])-stream.query(event[4]).numpy()).max()))
    outage = event[4]+2.                                                      # no slot within the span
    return max(worst, float(np.abs(runtime.query(outage)-stream.query(outage).numpy()).max()))


class DirectRuntimeTests(unittest.TestCase):
    def test_runtime_matches_stream(self):
        """Slot selection and ages in numpy around the networks equal the PyTorch stream."""
        model = with_outputs(HandDirect(direct()))
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/direct_runtime.META_FILE).write_text(json.dumps(dict(
                architecture='direct', export_format=direct_runtime.EXPORT_FORMAT, config=asdict(model.config))))
            runtime = direct_runtime.DirectRuntime(folder, graphs=TorchGraphs(model))
            self.assertLess(runtime_worst(model, runtime, moving_events(seed=3)), 1e-5)

    @unittest.skipIf(any(importlib.util.find_spec(name) is None for name in ('ncnn', 'pnnx')),
                     'Install requirements-export.txt for ncnn tests')
    def test_exported_ncnn_runtime_matches_stream(self):
        from hand_tracking.direct_export import export_direct
        model = with_outputs(HandDirect(direct()))
        with tempfile.TemporaryDirectory() as folder:
            export_direct(model, Path(folder)/'export')
            runtime = direct_runtime.DirectRuntime(Path(folder)/'export')
            self.assertLess(runtime_worst(model, runtime, moving_events(seed=3)), 2e-3)   # ncnn may use fp16


if __name__ == '__main__':
    unittest.main()
