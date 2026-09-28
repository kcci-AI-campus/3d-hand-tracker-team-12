"""HandLite: batch/stream equivalence, slot selection, numpy geometry, deployment runtime."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
from model_helpers import ORIGINS, asynchronous, synthetic_events
import torch
from hand_tracking.config import LiteConfig
from hand_tracking.checkpoints import build_model, load_checkpoint, save_checkpoint, stream_for
from hand_tracking.events import make_events, query_anchor, sample_points, latest_rays
from hand_tracking.geometry import triangulate
from hand_tracking import lite as lite_module
from hand_tracking.lite import HandLite, LiteStream
from hand_tracking import lite_runtime
from hand_tracking.objectives import pose_loss

EXPORT_MODULES = ('onnx', 'onnxscript', 'onnxruntime')


def lite(**overrides):
    return LiteConfig(**{**dict(dim=32, heads=4, blocks=1, dropout=0., slots_per_camera=4), **overrides})


def trained_looking(model, std=.05):
    """Nonzero residual heads and gap embedding, so outputs depend on every input."""
    heads = (model.decoder.output,)+tuple(head for head in (model.decoder.gap, model.calibrator) if head is not None)
    for head in (getattr(head, 'output', head) for head in heads):
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


SINGLE_VIEW_FROM = -1.


def single_view_events():
    """Two moving hands over 2 s; from SINGLE_VIEW_FROM on only camera 0 sees hand 0.
    Returns the inputs and the true position function."""
    torch.manual_seed(0)
    base = torch.randn(2,21,3)*.05+torch.tensor([[[-.25,0,1.]],[[.25,0,1.]]])
    velocity = torch.zeros(2,21,3)
    velocity[..., 0] = .4
    velocity[0, :, 2] = -.15                                          # hand 0 also changes depth
    position = lambda t: base+velocity*t
    captures, cameras = asynchronous(duration=2.)
    inputs = list(synthetic_events(position, captures, cameras))
    inputs[1] = inputs[1].clone()
    for i, (capture, camera) in enumerate(zip(inputs[3][0].tolist(), inputs[2][0].tolist())):
        if capture >= SINGLE_VIEW_FROM and camera:
            inputs[1][0, i, 0] = False
    return inputs, position


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

    def test_padding_slots_stay_excluded_whatever_the_time_bias(self):
        model = trained_looking(HandLite(lite(calibration_head=False)))
        with torch.no_grad():
            model.decoder.time_bias[-1].bias.fill_(1e6)                     # far above the -1e4 key mask
            model.decoder.gap[-1].bias.fill_(1e3)                           # must not leak into padding tokens
        inputs = moving_events()
        query = inputs[4][:, -1:]
        first = [value[:, -6:] for value in inputs]                         # 2 events per camera: padding slots
        other = [value.clone() for value in first]
        other[0][..., 5:8] = torch.nn.functional.normalize(torch.randn_like(other[0][..., 5:8]), dim=-1)
        # Padding slots point at event 0; a different event 0 outside every slot must not matter.
        outside = [torch.cat((a[:, :1], b), 1) for a, b in zip(other, first)]
        outside[3][:, 0] = outside[3][:, 1]-10.                              # captured long before the span
        outside[4][:, 0] = outside[4][:, 1]-9.99
        with torch.no_grad():
            torch.testing.assert_close(model(*outside, query), model(*first, query), atol=1e-6, rtol=1e-6)

    def test_stream_matches_forward(self):
        for calibration, fingers in ((True, True), (False, True), (True, False)):
            with self.subTest(calibration=calibration, finger_tokens=fingers):
                model = trained_looking(HandLite(lite(calibration_head=calibration, finger_tokens=fingers)))
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


class GapEmbeddingTests(unittest.TestCase):
    def test_starts_inert(self):
        torch.manual_seed(0)
        model = trained_looking(HandLite(lite()))
        with torch.no_grad():
            model.decoder.gap[-1].weight.zero_()
            model.decoder.gap[-1].bias.zero_()
        plain = HandLite(lite(gap_embedding=False))
        plain.load_state_dict({key: value for key, value in model.state_dict().items()
                               if not key.startswith('decoder.gap.')})
        inputs = moving_events()
        query = inputs[4][:, -1:]+torch.tensor([[0., .05]])
        with torch.no_grad():
            torch.testing.assert_close(model(*inputs, query), plain.eval()(*inputs, query))

    def test_token_values_depend_on_gap(self):
        """With attention fixed (constant time bias), only the gap embedding sees the gap."""
        torch.manual_seed(0)
        decoder = trained_looking(HandLite(lite())).decoder
        tokens, valid = torch.randn(1, 8, 32), torch.ones(1, 8)
        anchor = torch.randn(1, 42, LiteConfig().anchor_features)
        with torch.no_grad():
            decoder.time_bias[0].weight.zero_()                             # attention no longer sees the gap
            first = decoder(tokens, valid, torch.zeros(1, 8), anchor)
            later = decoder(tokens, valid, torch.full((1, 8), 2.), anchor)
            decoder.gap = None
            self.assertTrue(torch.equal(decoder(tokens, valid, torch.zeros(1, 8), anchor),
                                        decoder(tokens, valid, torch.full((1, 8), 2.), anchor)))
        self.assertGreater(float((first-later).abs().max()), 1e-4)



class FingerTokenTests(unittest.TestCase):
    def test_each_finger_token_sees_only_its_joints(self):
        torch.manual_seed(0)
        encoder = HandLite(lite()).encoder.eval()
        features, camera = torch.randn(1, 2, 21, 14), torch.eye(3)[:1]
        encode = lambda x: encoder(x.reshape(1, 2, -1), camera).reshape(2, 5, -1)
        base = encode(features)
        for hand, joint, changed in ((0, 6, {(0, 1)}), (1, 20, {(1, 4)}), (1, 0, {(1, f) for f in range(5)})):
            moved = features.clone()
            moved[0, hand, joint] += 1.
            difference = (encode(moved)-base).abs().amax(-1)
            self.assertEqual({tuple(i) for i in (difference > 1e-6).nonzero().tolist()}, changed, (hand, joint))

    def test_token_count(self):
        inputs = moving_events()
        for fingers, count in ((True, 10), (False, 2)):
            model = HandLite(lite(finger_tokens=fingers))
            self.assertEqual(model.config.event_tokens, count)
            encoded = model.encode_events(make_events(*inputs))
            self.assertEqual(encoded.tokens.shape[2], count)


class LegacyTests(unittest.TestCase):
    """Checkpoints and exports written before gap_embedding/finger_tokens existed."""

    def test_old_checkpoints_load_without_new_options(self):
        old = HandLite(lite(gap_embedding=False, finger_tokens=False))
        config = vars(old.config).copy()
        del config['gap_embedding'], config['finger_tokens']                  # written before the options existed
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'old.pt'
            save_checkpoint(path, dict(model=old.state_dict(), architecture='lite', model_config=config))
            loaded = load_checkpoint(path).model
        self.assertIsNone(loaded.decoder.gap)
        self.assertEqual(loaded.config.event_tokens, 2)
        self.assertTrue(HandLite().config.gap_embedding and HandLite().config.finger_tokens)

    def test_old_exports_load_with_one_token_per_hand(self):
        import json
        config = vars(lite()).copy()
        del config['gap_embedding'], config['finger_tokens']
        with tempfile.TemporaryDirectory() as folder:
            meta = Path(folder)/lite_runtime.META_FILE
            for version, readable in ((1, False), (2, True), (3, True)):
                meta.write_text(json.dumps(dict(architecture='lite', export_format=version, config=config)))
                if not readable:
                    with self.assertRaises(ValueError):
                        lite_runtime.LiteRuntime(folder, graphs=object())
                    continue
                runtime = lite_runtime.LiteRuntime(folder, graphs=object())
                self.assertEqual(runtime.config.event_tokens, 2)


class SingleRayAnchorTests(unittest.TestCase):
    def test_single_view_hand_stays_on_its_ray(self):
        inputs, position = single_view_events()
        ray, old = HandLite(lite()).eval(), HandLite(lite(anchor_ray_depth=False)).eval()   # pose == anchor
        for after in (.29, .5, .9):                                        # beyond anchor_hold_s
            query = SINGLE_VIEW_FROM+after
            count = int((inputs[4][0] <= query).sum())
            window = [value[:, :count] for value in inputs]
            with torch.no_grad():
                pose, old_pose = (model(*window, torch.tensor([[query]]))[0, 0] for model in (ray, old))
            truth = position(query)
            error = lambda value, hand: float((value[hand]-truth[hand]).norm(dim=-1).mean())
            self.assertGreater(error(old_pose, 0), .8, after)                # old rule: hand 0 at the origin
            self.assertLess(error(pose, 0), .15, after)                      # depth drift only
            self.assertLess(error(pose, 1), .02, after)                      # the other hand is untouched
            # Every hand-0 joint lies on camera 0's ray through its true position at that frame.
            direction = torch.nn.functional.normalize(pose[0]-ORIGINS[0], dim=-1)
            last = inputs[3][0][(inputs[2][0] == 0) & (inputs[4][0] <= query)].max()
            seen = torch.nn.functional.normalize(position(float(last))[0]-ORIGINS[0], dim=-1)
            self.assertLess(float((direction-seen).norm(dim=-1).max()), 1e-4, after)

    def test_flag_and_memory(self):
        inputs, _ = single_view_events()
        model = HandLite(lite()).eval()
        events = make_events(*inputs)
        encoded = model.encode_events(events)
        flags = []
        for query in (SINGLE_VIEW_FROM-.1, SINGLE_VIEW_FROM+.5, SINGLE_VIEW_FROM+1.4):
            captured = {}
            original = lite_module.query_anchor
            def spy(*args, **kwargs):
                captured['flags'] = original(*args, **kwargs)[1]
                return original(*args, **kwargs)
            lite_module.query_anchor = spy
            try:
                with torch.no_grad():
                    model.decode_queries(events, encoded, torch.tensor([[query]]))
            finally:
                lite_module.query_anchor = original
            flags.append(captured['flags'][0, 0, :, :, 5])                   # [2,21] single-ray flag
        self.assertFalse(bool(flags[0].any()))                               # both hands triangulated
        self.assertTrue(bool(flags[1][0].all()) and not bool(flags[1][1].any()))
        # 1.4 s after the last triangulation the remembered depth (1 s) has expired.
        self.assertFalse(bool(flags[2][0].any()))


class OutlierRayTests(unittest.TestCase):
    """robust_triangulate drops one ray of three that the other two contradict."""

    def rays(self, points, cameras=ORIGINS):
        """Exact rays [...,3,3] from each camera to points [...,3]."""
        direction = torch.nn.functional.normalize(points[..., None, :]-cameras, dim=-1)
        return cameras.expand_as(direction), direction

    def test_drops_the_contradicted_ray_only(self):
        from hand_tracking.geometry import robust_triangulate
        torch.manual_seed(0)
        truth = torch.randn(200, 3)*.1+torch.tensor([0., 0., 1.])
        origins, directions = self.rays(truth)
        mask = torch.ones(200, 3, dtype=torch.bool)
        noisy = torch.nn.functional.normalize(directions+torch.randn(directions.shape)*.002, dim=-1)
        point, ok, used = robust_triangulate(origins, noisy, mask, 6.)
        self.assertTrue(bool(ok.all()) and bool(used.all()))                   # consistent: nothing dropped
        wrong = noisy.clone()
        wrong[:, 1] = self.rays(truth+torch.tensor([0., 0., .5]))[1][:, 1]     # camera 1 sees another point
        point, ok, used = robust_triangulate(origins, wrong, mask, 6.)
        self.assertTrue(bool(ok.all()))
        self.assertTrue(bool((used == torch.tensor([True, False, True])).all()))
        self.assertLess(float((point-truth).norm(dim=-1).max()), .03)            # a two-ray point
        plain, _ = triangulate(origins, wrong, mask)
        self.assertGreater(float((plain-truth).norm(dim=-1).mean()), .1)       # without the test: pulled away
        # Displaced along x, within the epipolar plane of cameras 0 and 1 (both at y=-1, z=.5):
        # (0,1) and (0,2) agree, (1,2) do not; camera 1 or 2 may be wrong, so no sample.
        epipolar = noisy.clone()
        epipolar[:, 1] = self.rays(truth+torch.tensor([.5, 0., 0.]))[1][:, 1]
        point, ok, used = robust_triangulate(origins, epipolar, mask, 6.)
        self.assertLess(float(ok.float().mean()), .05)
        two = mask.clone()
        two[:, 2] = False                                                      # two rays: nothing to vote with
        point, ok, used = robust_triangulate(origins, wrong, two, 6.)
        self.assertTrue(bool((used == two).all()))

    def test_numpy_matches_torch(self):
        from hand_tracking.geometry import robust_triangulate
        torch.manual_seed(1)
        truth = torch.randn(500, 3)*.1+torch.tensor([0., 0., 1.])
        origins, directions = self.rays(truth)
        directions = torch.nn.functional.normalize(directions+torch.randn(directions.shape)*.01, dim=-1)
        outlier = torch.rand(500) < .3
        directions[outlier, 0] = torch.nn.functional.normalize(torch.randn(int(outlier.sum()), 3), dim=-1)
        mask = torch.rand(500, 3) > .15
        point, ok, used = robust_triangulate(origins.double(), directions.double(), mask, 6.)
        n_point, n_ok, n_used = lite_runtime.robust_triangulate(origins.double().numpy(), directions.double().numpy(),
                                                                mask.numpy(), 6.)
        np.testing.assert_array_equal(n_ok, ok.numpy())
        np.testing.assert_array_equal(n_used, used.numpy())
        np.testing.assert_allclose(n_point, point.numpy(), atol=1e-9)
        self.assertGreater(int((used != mask).sum()), 50)                      # the test did drop rays

    def test_swapped_camera(self):
        """Camera 1 reports hand 1 as hand 0 for three frames (hands 0.5 apart). Samples the
        test accepts equal those without camera 1's hand 0; the anchor error drops by far."""
        torch.manual_seed(0)
        base = torch.randn(2,21,3)*.05
        base[1] += torch.tensor([0., .3, .4])
        velocity = torch.randn(2,21,3)*.1+torch.randn(2,1,3)*.3
        clean = list(synthetic_events(lambda t: base+velocity*t, *asynchronous(duration=1.2)))
        swapped, missing = [value.clone() for value in clean], [value.clone() for value in clean]
        for i in torch.nonzero(clean[2][0] == 1)[-3:, 0].tolist():
            swapped[0][0, i, 0] = clean[0][0, i, 1]
            missing[1][0, i, 0] = False
        model = HandLite(lite()).eval()                                       # pose == anchor
        samples = {}
        for name, inputs, ratio in (('swapped', swapped, 6.), ('missing', missing, 0.)):
            encoded = model.encode_events(make_events(*inputs))
            samples[name] = sample_points(encoded.origin, encoded.direction, encoded.mask, encoded.capture,
                                          outlier_ratio=ratio)
        accepted = samples['swapped'][1]
        self.assertFalse(bool((accepted & ~samples['missing'][1]).any()))
        torch.testing.assert_close(samples['swapped'][0][accepted], samples['missing'][0][accepted], atol=1e-5, rtol=0)
        query = clean[4][:, -1:]
        with torch.no_grad():
            truth = HandLite(lite(ray_outlier_ratio=0.)).eval()(*clean, query)[0, 0, 0]
            error = {ratio: float((HandLite(lite(ray_outlier_ratio=ratio)).eval()(*swapped, query)[0, 0, 0]-truth)
                                  .norm(dim=-1).mean()) for ratio in (0., 6.)}
        self.assertLess(error[6.], error[0.]/4)


class SpeedPathTests(unittest.TestCase):
    """Shortcuts that must not change results."""

    def test_anchor_geometry_skips_only_unreachable_slots(self):
        for inputs in (moving_events(), single_view_events()[0]):
            model = trained_looking(HandLite(lite(ray_outlier_ratio=6.)))
            query = inputs[4][:, -1:]+torch.tensor([[0., .05, .2]])
            with torch.no_grad():
                fast = model(*inputs, query, return_details=True)
                model.reaching_slots = lambda valid, arrival, query: valid.shape[-1]      # every slot
                full = model(*inputs, query, return_details=True)
            torch.testing.assert_close(fast.pose, full.pose, atol=1e-5, rtol=0)
            torch.testing.assert_close(fast.calibration, full.calibration)

    def test_reaching_slots_is_shorter_than_all(self):
        model = HandLite(lite(slots_per_camera=8))
        inputs = moving_events(seconds=1.2)
        events = make_events(*inputs)
        query = inputs[4][:, -1:]
        slot, valid = model.select(events, query)
        arrival = events['arrival'][0][slot[0]]
        self.assertLess(model.reaching_slots(valid[0], arrival, query[0][:, None]), valid.shape[-1])

    def test_attention_dropout_is_separate(self):
        model = HandLite(lite(dropout=.1))
        self.assertEqual(model.decoder.blocks[0].cross.dropout, 0.)
        self.assertEqual(model.decoder.blocks[0].drop.p, .1)
        model = HandLite(lite(dropout=.1, attention_dropout=.2))
        self.assertEqual(model.calibrator.pool.dropout, .2)

    def test_compile_keeps_checkpoint_names(self):
        from training.train import compile_networks
        model = HandLite(lite())
        names = list(model.state_dict())
        compile_networks(model)
        self.assertEqual(list(model.state_dict()), names)


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
            # Camera outages (no slot within the span, then one camera back), short and very long.
            outage = event[4]+2.
            for query in (outage, outage+.1, outage+1e5):
                if query != outage:
                    for runner in (stream, *runtimes.values()):
                        runner.push(0, event[1], event[2], query-.05, query)
                expected = stream.query(query).numpy()
                for name, runtime in runtimes.items():
                    worst[name] = max(worst[name], float(np.abs(runtime.query(query)-expected).max()))
        self.assertLess(worst['onnxruntime'], 1e-4)
        self.assertTrue(np.isfinite(list(worst.values())).all())
        if 'ncnn' in worst:
            self.assertLess(worst['ncnn'], 2e-3)                              # ncnn may use fp16


class NcnnInputTests(unittest.TestCase):
    def test_float64_inputs_stay_alive_until_copied(self):
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


class SingleViewRuntimeTests(unittest.TestCase):
    def test_runtime_matches_stream_on_single_view(self):
        from hand_tracking.lite_export import export_lite
        if any(importlib.util.find_spec(name) is None for name in EXPORT_MODULES):
            self.skipTest('Install requirements-export.txt for runtime tests')
        model = trained_looking(HandLite(lite()))
        inputs, _ = single_view_events()
        with tempfile.TemporaryDirectory() as folder:
            export_lite(model, Path(folder)/'export', ['onnx'])
            runtime = lite_runtime.LiteRuntime(Path(folder)/'export', backend='onnxruntime')
            stream = LiteStream(model)
            worst, rays = 0., 0
            for i in range(inputs[0].shape[1]):
                event = (int(inputs[2][0,i]), inputs[0][0,i].numpy(), inputs[1][0,i].numpy(),
                         float(inputs[3][0,i]), float(inputs[4][0,i]))
                runtime.push(*event)
                stream.push(*event)
                if i % 3 == 2:
                    pose, _ = runtime.query_details(event[4]+.01)
                    worst = max(worst, float(np.abs(pose-stream.query(event[4]+.01).numpy()).max()))
                    rays += event[3] > SINGLE_VIEW_FROM+.35
        self.assertGreater(rays, 5)
        self.assertLess(worst, 1e-4)


class RuntimeInputTests(unittest.TestCase):
    def test_runtime_owns_reused_numpy_buffers(self):
        from hand_tracking.lite_export import export_lite
        if importlib.util.find_spec('onnxruntime') is None or importlib.util.find_spec('onnxscript') is None:
            self.skipTest('Install requirements-export.txt for runtime tests')
        model = trained_looking(HandLite(lite()))
        inputs = moving_events(seed=5)
        with tempfile.TemporaryDirectory() as folder:
            export_lite(model, Path(folder)/'export', ['onnx'])
            reference = lite_runtime.LiteRuntime(Path(folder)/'export', backend='onnxruntime')
            reused = lite_runtime.LiteRuntime(Path(folder)/'export', backend='onnxruntime')
            features, valid = np.empty((2,21,14), np.float32), np.empty((2,21), bool)
            for camera in (-1, 3):                                            # -1 must not alias camera 2
                with self.assertRaises(ValueError):
                    reused.push(camera, features, valid, 0., .01)
            for i in range(inputs[0].shape[1]):
                event = (int(inputs[2][0,i]), float(inputs[3][0,i]), float(inputs[4][0,i]))
                reference.push(event[0], inputs[0][0,i].numpy().copy(), inputs[1][0,i].numpy().copy(), *event[1:])
                np.copyto(features, inputs[0][0,i].numpy())
                np.copyto(valid, inputs[1][0,i].numpy())
                reused.push(event[0], features, valid, *event[1:])
                features.fill(np.nan)
                valid.fill(False)                                            # caller reuses its buffers
                np.testing.assert_array_equal(reused.query(event[2]), reference.query(event[2]))


class StreamInputTests(unittest.TestCase):
    def test_out_of_range_camera_is_rejected_everywhere(self):
        features, valid = np.zeros((2,21,14), np.float32), np.ones((2,21), bool)
        streams = [stream_for(HandLite(lite()))]
        for stream in streams:
            for camera in (-1, 3, 1.5):
                with self.subTest(stream=type(stream).__name__, camera=camera), self.assertRaises(ValueError):
                    stream.push(camera, features, valid, 0., .01)
            with self.assertRaises(ValueError):
                stream.push(0, features[:1], valid, 0., .01)
            self.assertEqual(len(stream), 0)


class ExportValidationTests(unittest.TestCase):
    def test_non_finite_poses_fail_validation(self):
        from training import export as export_cli
        good = np.zeros((2,21,3))
        bad = good.copy()
        bad[0,0,0] = np.nan
        self.assertEqual(export_cli.pose_difference(good+.5, good, 'onnx', 0.), .5)
        for actual, expected in ((bad, good), (good, bad)):
            with self.assertRaises(FloatingPointError):
                export_cli.pose_difference(actual, expected, 'onnx', 0.)
        with self.assertRaises(SystemExit):
            export_cli.parse_args(['--checkpoint', 'x', '--output', 'y', '--queries', '0'])


class CheckpointTests(unittest.TestCase):
    def test_build_model_by_architecture(self):
        self.assertIsInstance(build_model('lite', lite()), HandLite)
        with self.assertRaises(KeyError):
            build_model('unknown', lite())


if __name__ == '__main__':
    unittest.main()
