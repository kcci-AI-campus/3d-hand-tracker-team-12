"""HandLite: batch/stream equivalence, slot selection, numpy geometry, deployment runtime."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
from model_helpers import synthetic_events
import torch
from hand_tracking.config import LiteConfig
from hand_tracking.checkpoints import build_model, stream_for
from hand_tracking.events import make_events, query_anchor, sample_points, latest_rays
from hand_tracking.lite import HandLite, LiteStream
from hand_tracking import lite_runtime
from hand_tracking.objectives import pose_loss

EXPORT_MODULES = ('onnx', 'onnxscript', 'onnxruntime')


def lite(**overrides):
    return LiteConfig(**{**dict(dim=32, heads=4, blocks=1, dropout=0., slots_per_camera=4), **overrides})


def trained_looking(model, std=.05):
    """Nonzero residual heads, so outputs depend on every input."""
    for head in (model.decoder.output,)+((model.calibrator.output,) if model.calibrator is not None else ()):
        for parameter in head[-1].parameters():
            torch.nn.init.normal_(parameter, std=std)
    return model.eval()


def moving_events(seconds=1.2, rate=17.1, seed=0):
    torch.manual_seed(seed)
    base, velocity = torch.randn(2,21,3)*.1, torch.randn(2,21,3)*.3
    times = np.arange(-seconds, 0, 1/rate)
    inputs = list(synthetic_events(lambda t: base+velocity*t, [t for t in times for _ in range(3)],
                                   [c for _ in times for c in range(3)]))
    inputs[1] = inputs[1].clone()
    inputs[1][0, ::5, 1, :7] = False                                  # some missing joints
    return inputs


class LiteModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shapes_padding_backward_and_causality(self):
        model = trained_looking(HandLite(lite()))
        inputs = moving_events()
        query = torch.stack((inputs[4][:,-7], inputs[4][:,-1]), 1)
        output = model(*inputs, query, return_details=True)
        self.assertEqual((output.pose.shape, output.calibration.shape), ((1,2,2,21,3), (1,2,3,6)))
        self.assertTrue(output.accepted.all() and torch.isfinite(output.pose).all())
        padded = [torch.cat((value, value[:,:5]), 1) for value in inputs]
        padded[0][:,-5:] = float('nan')
        padded[5][:,-5:] = False
        torch.testing.assert_close(model(*padded, query), output.pose, atol=1e-5, rtol=1e-5)
        altered = [value.clone() for value in inputs]
        altered[0][0,-6:,...,5:8] = torch.nn.functional.normalize(torch.randn(6,2,21,3), dim=-1)
        with torch.no_grad():
            changed = model(*altered, query)
        torch.testing.assert_close(changed[:,0], output.pose[:,0].detach())   # later arrivals cannot matter
        self.assertFalse(torch.allclose(changed[:,1], output.pose[:,1]))
        model.train()
        loss = pose_loss(model(*inputs, query), torch.zeros(1,2,2,21,3), torch.ones(1,2,2,21, dtype=torch.bool),
                         torch.tensor([30.]))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertGreater(model.calibrator.output[-1].weight.grad.abs().sum().item(), 0)

    def test_slots_are_each_cameras_latest_events_in_arrival_order(self):
        model = HandLite(lite(slots_per_camera=2, event_span_s=.6))
        inputs = synthetic_events(lambda t: torch.zeros(2,21,3), [-.5,-.3,-.25,-.2,-.15,-.1,-.05], [0,0,1,0,1,0,2])
        events = make_events(*inputs)
        query = inputs[4][:, -1:]                                           # after every arrival
        slot, valid = model.select(events, query)
        chosen = sorted(int(i) for i, v in zip(slot[0,0], valid[0,0]) if v)
        # All within the span; K=2 keeps camera 0's latest two (-.2, -.1), not -.5 or -.3.
        cameras, captures = inputs[2][0][chosen].tolist(), [round(float(c), 2) for c in inputs[3][0][chosen]]
        self.assertEqual(sorted(zip(cameras, captures)), [(0,-.2), (0,-.1), (1,-.25), (1,-.15), (2,-.05)])
        self.assertEqual(int((~valid).sum()), 1)                          # one padding slot, placed first
        self.assertFalse(bool(valid[0,0,0]))

    def test_stream_matches_forward(self):
        for calibration in (True, False):
            with self.subTest(calibration=calibration):
                model = trained_looking(HandLite(lite(calibration_head=calibration)))
                inputs = moving_events()
                stream = stream_for(model)
                self.assertIsInstance(stream, LiteStream)
                epoch = 1_700_000_000.
                for i in range(inputs[0].shape[1]):
                    stream.push(int(inputs[2][0,i]), inputs[0][0,i], inputs[1][0,i],
                                epoch+float(inputs[3][0,i]), epoch+float(inputs[4][0,i]))
                    if i % 4 == 3:
                        time = float(inputs[4][0,i])+.013
                        with torch.no_grad():
                            expected = model(*[value[:, :i+1] for value in inputs], torch.tensor([[time]]))[0,0]
                        torch.testing.assert_close(stream.query(epoch+time), expected, atol=1e-4, rtol=1e-4)


class NumpyGeometryTests(unittest.TestCase):
    """lite_runtime's numpy geometry against the torch implementation."""

    def test_sample_and_anchor_match_torch(self):
        inputs = moving_events()
        events = make_events(*inputs)
        targets = torch.arange(inputs[0].shape[1])
        origin, direction, mask, capture = latest_rays(events, targets, .2)
        params = torch.randn(3,6)*torch.tensor([.05]*3+[.05]*3)
        expand = params[None, None].expand(1, len(targets), -1, -1)
        point, ok, stamp, _ = sample_points(origin, direction, mask, capture, expand)
        n_point, n_ok, n_stamp = lite_runtime.sample(origin[0].double().numpy(), direction[0].double().numpy(),
                                                     mask[0].numpy(), capture[0].double().numpy(), params.double().numpy())
        np.testing.assert_array_equal(n_ok, ok[0].numpy())
        np.testing.assert_allclose(n_point, point[0].numpy(), atol=2e-5)
        np.testing.assert_allclose(n_stamp, stamp[0].numpy(), atol=1e-6)
        query = float(inputs[4][0,-1])+.02
        anchor, flags = query_anchor(point, ok, stamp, events['arrival'], events['present'], torch.tensor([[query]]), .2, .3)
        n_anchor, n_flags = lite_runtime.anchor_at(n_point, n_ok, n_stamp, query, .2, .3)
        np.testing.assert_allclose(n_anchor, anchor[0,0].numpy(), atol=5e-5)
        np.testing.assert_allclose(n_flags, flags[0,0].numpy(), atol=1e-4)

    def test_rodrigues_matches_torch(self):
        from hand_tracking.geometry import rodrigues
        w = np.random.default_rng(0).normal(size=(50,3))*np.logspace(-6, 0, 50)[:,None]
        np.testing.assert_allclose(lite_runtime.rodrigues(w), rodrigues(torch.tensor(w)).numpy(), atol=1e-7)


@unittest.skipIf(any(importlib.util.find_spec(name) is None for name in EXPORT_MODULES),
                 'Install requirements-export.txt for runtime tests')
class RuntimeTests(unittest.TestCase):
    def test_exported_runtime_matches_stream(self):
        from hand_tracking.lite_export import export_lite
        backends = ['onnxruntime'] + (['ncnn'] if all(importlib.util.find_spec(n) for n in ('ncnn', 'pnnx')) else [])
        formats = ['onnx'] + (['ncnn'] if 'ncnn' in backends else [])
        model = trained_looking(HandLite(lite()))
        inputs = moving_events(seed=3)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)/'export'
            export_lite(model, directory, formats)
            runtimes = {name: lite_runtime.LiteRuntime(directory, backend=name) for name in backends}
            stream = LiteStream(model)
            worst = dict.fromkeys(runtimes, 0.)
            for i in range(inputs[0].shape[1]):
                event = (int(inputs[2][0,i]), inputs[0][0,i].numpy(), inputs[1][0,i].numpy(),
                         float(inputs[3][0,i]), float(inputs[4][0,i]))
                self.assertEqual({runtime.push(*event) for runtime in runtimes.values()}, {stream.push(*event)})
                if i % 5 == 4:
                    expected = stream.query(event[4]).numpy()
                    for name, runtime in runtimes.items():
                        worst[name] = max(worst[name], float(np.abs(runtime.query(event[4])-expected).max()))
        self.assertLess(worst['onnxruntime'], 1e-4)
        if 'ncnn' in worst:
            self.assertLess(worst['ncnn'], 2e-3)                              # ncnn may use fp16


class CheckpointTests(unittest.TestCase):
    def test_build_model_by_architecture(self):
        self.assertIsInstance(build_model('lite', lite()), HandLite)
        with self.assertRaises(KeyError):
            build_model('unknown', lite())


if __name__ == '__main__':
    unittest.main()
