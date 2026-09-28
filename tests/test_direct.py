"""HandDirect: networks only; batch/stream equivalence and the deployment runtime."""
import importlib.util
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

EXPORT_MODULES = ('onnx', 'onnxscript', 'onnxruntime')


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
        self.assertIsInstance(build_model('direct', direct()), HandDirect)
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


@unittest.skipIf(any(importlib.util.find_spec(name) is None for name in EXPORT_MODULES),
                 'Install requirements-export.txt for runtime tests')
class DirectRuntimeTests(unittest.TestCase):
    def test_exported_runtime_matches_stream(self):
        from hand_tracking.direct_export import export_direct
        backends = ['onnxruntime'] + (['ncnn'] if all(importlib.util.find_spec(n) for n in ('ncnn', 'pnnx')) else [])
        formats = ['onnx'] + (['ncnn'] if 'ncnn' in backends else [])
        model = with_outputs(HandDirect(direct()))
        inputs = moving_events(seed=3)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)/'export'
            export_direct(model, directory, formats)
            runtimes = {name: direct_runtime.DirectRuntime(directory, backend=name) for name in backends}
            stream = DirectStream(model)
            worst = dict.fromkeys(runtimes, 0.)
            for i in range(inputs[0].shape[1]):
                event = (int(inputs[2][0,i]), inputs[0][0,i].numpy(), inputs[1][0,i].numpy(),
                         float(inputs[3][0,i]), float(inputs[4][0,i]))
                self.assertEqual({runtime.push(*event) for runtime in runtimes.values()}, {stream.push(*event)})
                if i % 5 == 4:
                    expected = stream.query(event[4]).numpy()
                    for name, runtime in runtimes.items():
                        worst[name] = max(worst[name], float(np.abs(runtime.query(event[4])-expected).max()))
            outage = event[4]+2.                                                  # no slot within the span
            expected = stream.query(outage).numpy()
            for name, runtime in runtimes.items():
                worst[name] = max(worst[name], float(np.abs(runtime.query(outage)-expected).max()))
        self.assertLess(worst['onnxruntime'], 1e-4)
        if 'ncnn' in worst:
            self.assertLess(worst['ncnn'], 2e-3)                              # ncnn may use fp16


if __name__ == '__main__':
    unittest.main()
