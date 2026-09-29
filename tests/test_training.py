"""hand_tracking.engine, checkpoints and config: metrics, checkpoint round trips, CLI generation."""
import argparse
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from model_helpers import small, small_direct
import torch
from hand_tracking.checkpoints import (build_model, load_checkpoint, restore_training, save_checkpoint,
                                       training_checkpoint)
from hand_tracking.config import (DirectConfig, LiteV3Config, SamplingConfig, add_model_arguments,
                                  model_config_from_args)
from hand_tracking.contracts import ModelOutput
from hand_tracking.engine import run_epoch
from hand_tracking.events import accepted_captures

CPU = torch.device('cpu')


class ZeroModel(torch.nn.Module):
    """Predicts zero pose, zero expected error (log(1 + 0 mm)) and 'in view' for every hand, so
    metrics reduce to the targets."""
    config = LiteV3Config()

    def forward(self, features, valid, camera, capture, arrival, present, query, return_details=False):
        b, q = query.shape
        pose = torch.zeros(b, q, 2, 21, 3)
        return ModelOutput(pose, None, accepted_captures(camera, capture, present), error=torch.zeros(b, q, 2, 21),
                           presence=torch.full((b, q, 2), 5.))


def zero_batch(b=1, events=2):
    return dict(event_features=torch.zeros(b,events,2,21,14), event_valid=torch.ones(b,events,2,21, dtype=torch.bool),
                event_camera=torch.zeros(b,events, dtype=torch.long),
                event_capture=(torch.arange(events).float()*.01-.1).expand(b,-1),
                event_arrival=(torch.arange(events).float()*.01-.09).expand(b,-1),
                event_present=torch.ones(b,events, dtype=torch.bool), query_times=torch.zeros(b,2),
                target=torch.zeros(b,2,2,21,3), target_world=torch.zeros(b,2,2,21,3),
                target_mask=torch.ones(b,2,2,21, dtype=torch.bool), hand_in_view=torch.ones(b,2,2, dtype=torch.bool),
                world_unit_cm=torch.full((b,), 40.), calibration_target=torch.full((b,3,6), float('nan')))


class MetricTests(unittest.TestCase):
    def test_final_query_metrics(self):
        batch = zero_batch()
        batch['target'][:,-1,...,0] = .1                                      # 4 cm at 40 cm per unit
        batch['target_world'][:,-1,...,1] = .2                                # 8 cm off in the world frame
        report = run_epoch(ZeroModel(), [batch], CPU)
        self.assertAlmostEqual(report['mpjpe_mm'], 40., places=4)
        self.assertAlmostEqual(report['mpjpe_world_mm'], 80., places=4)
        self.assertAlmostEqual(report['mpjpe_rel_mm'], 0., places=4)          # same shift for every joint
        self.assertAlmostEqual(report['error_miss_mm'], 40., places=3)        # expected 0 mm, actual 40 mm
        self.assertEqual((report['pck20'], report['samples'], report['joints'], report['batches']), (0., 1, 42, 1))
        self.assertIsNone(report['kind_rate']['triangulated'])                # no anchor output

    def test_in_view_accuracy_and_rate(self):
        batch = zero_batch()
        batch['hand_in_view'][:, -1, 1] = False                               # the model says 'in view'
        batch['target_mask'][:, -1, 1] = False
        report = run_epoch(ZeroModel(), [batch], CPU)
        self.assertAlmostEqual(report['presence_accuracy'], .5)
        self.assertAlmostEqual(report['out_of_view_rate'], .5)
        self.assertEqual(report['joints'], 21)                                # the out-of-view hand has no target


class ReprojectionTests(unittest.TestCase):
    class MovingHands(torch.nn.Module):
        """Predicts the true moving hands at any query time, plus a learnable offset."""
        def __init__(self, base, velocity):
            super().__init__()
            self.base, self.velocity = base, velocity
            self.offset = torch.nn.Parameter(torch.zeros(3))

        def forward(self, features, valid, camera, capture, arrival, present, query, return_details=False):
            return self.base+self.velocity*query[..., None, None, None]+self.offset

    def batch(self):
        from model_helpers import synthetic_events
        torch.manual_seed(0)
        base, velocity = torch.randn(2, 21, 3)*.1, torch.randn(2, 21, 3)*.3
        captures = [t for t in torch.arange(-.6, 0, .085).tolist() for _ in range(3)]
        inputs = synthetic_events(lambda t: base+velocity*t, captures, [c for _ in range(len(captures)//3) for c in range(3)])
        keys = ('event_features', 'event_valid', 'event_camera', 'event_capture', 'event_arrival', 'event_present')
        batch = dict(zip(keys, inputs), query_times=torch.zeros(1, 1))
        return batch, self.MovingHands(base, velocity)

    def test_exact_pose_has_no_reprojection_error_and_an_offset_does(self):
        from hand_tracking.engine import reprojection_loss
        batch, model = self.batch()
        loss, angle, count = reprojection_loss(model, batch, 4)
        self.assertEqual(int(count), 4*42)                                      # four frames, every joint detected
        self.assertLess(float(angle/count), 1e-5)
        self.assertLess(float(loss), 1e-6)
        with torch.no_grad():
            model.offset += torch.tensor([.05, 0., 0.])                          # 2 cm off
        loss, angle, count = reprojection_loss(model, batch, 4)
        self.assertGreater(float(angle/count), .005)
        loss.backward()
        self.assertGreater(float(model.offset.grad.abs().sum()), 0.)

    def test_checks_the_newest_detecting_frames_only(self):
        from hand_tracking.engine import reprojection_loss
        batch, model = self.batch()
        batch['event_valid'] = batch['event_valid'].clone()
        batch['event_valid'][0, -3:] = False                                     # the newest three detect nothing
        seen = []
        forward = model.forward
        def record(*args, **kwargs):
            seen.append(args[6].clone())
            return forward(*args, **kwargs)
        model.forward = record
        _, _, count = reprojection_loss(model, batch, 4)
        self.assertEqual(int(count), 4*42)
        capture = batch['event_capture'][0]
        torch.testing.assert_close(seen[0][0].sort().values, capture[-7:-3].sort().values)


class OptimizerStepTests(unittest.TestCase):
    """Loss finite, gradient infinite (sqrt at 0): skipped with mixed precision, an error without."""

    class Scaler:
        """A GradScaler stand-in that is enabled on CPU."""
        def __init__(self):
            self.stepped = False
        def is_enabled(self):
            return True
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            if all(torch.isfinite(p.grad).all() for g in optimizer.param_groups for p in g['params']):
                self.stepped = True
                optimizer.step()
        def update(self):
            pass

    def test_overflow_is_skipped_only_with_mixed_precision(self):
        from hand_tracking.engine import optimizer_step
        weight = torch.nn.Parameter(torch.zeros(1))
        model = torch.nn.Module()
        model.weight = weight
        optimizer = torch.optim.SGD([weight], lr=.1)
        loss = lambda: weight.abs().sqrt().sum()
        scaler = self.Scaler()
        self.assertFalse(optimizer_step(model, optimizer, scaler, loss()))
        self.assertFalse(scaler.stepped)
        self.assertEqual(float(weight.detach()), 0.)
        with self.assertRaises(RuntimeError):
            optimizer_step(model, optimizer, torch.amp.GradScaler('cuda', enabled=False), loss())


class CheckpointTests(unittest.TestCase):
    def test_each_architecture_round_trips(self):
        for config in (small(), small_direct()):
            with self.subTest(type(config).__name__), tempfile.TemporaryDirectory() as folder:
                model = build_model(config).eval()
                path = Path(folder)/'best.pt'
                optimizer = torch.optim.AdamW(model.parameters())
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
                save_checkpoint(path, training_checkpoint(model, SamplingConfig(37), optimizer, scheduler,
                                                          torch.amp.GradScaler('cuda', enabled=False), 0, 1., {}))
                restored = load_checkpoint(path)
                self.assertEqual(restored.model.config, config)
                self.assertEqual(restored.sampling.max_events, 37)
                self.assertFalse(restored.model.training)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(restored.model.state_dict()[name], value, atol=0, rtol=0)

    def test_unknown_architecture_is_rejected(self):
        model = build_model(small())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'old.pt'
            torch.save(dict(model=model.state_dict(), architecture='lite', model_config=asdict(model.config)), path)
            with self.assertRaisesRegex(ValueError, 'lite'):
                load_checkpoint(path)

    def test_training_state_round_trip(self):
        def objects(seed):
            torch.manual_seed(seed)
            model = build_model(small())
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
            return model, optimizer, scheduler, torch.amp.GradScaler('cuda', enabled=False)
        model, optimizer, scheduler, scaler = objects(0)
        model(*[torch.zeros(1,1,2,21,14), torch.zeros(1,1,2,21, dtype=torch.bool), torch.zeros(1,1, dtype=torch.long),
                torch.zeros(1,1), torch.zeros(1,1), torch.ones(1,1, dtype=torch.bool)], torch.zeros(1,1)).sum().backward()
        optimizer.step()
        scheduler.step()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'last.pt'
            save_checkpoint(path, training_checkpoint(model, SamplingConfig(64), optimizer, scheduler, scaler,
                                                      epoch=2, best_val_mpjpe_mm=12.5, options=dict(seed=0)))
            self.assertEqual(load_checkpoint(path).sampling.max_events, 64)
            restored = objects(1)
            self.assertEqual(restore_training(path, *restored), (3, 12.5))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(restored[0].state_dict()[name], value, atol=0, rtol=0)
        self.assertEqual(restored[2].state_dict(), scheduler.state_dict())


class ConfigCliTests(unittest.TestCase):
    def parse(self, config_class, *argv):
        parser = argparse.ArgumentParser()
        add_model_arguments(parser, config_class)
        return model_config_from_args(parser.parse_args(list(argv)), config_class)

    def test_defaults_round_trip(self):
        self.assertEqual(self.parse(LiteV3Config), LiteV3Config())
        self.assertEqual(self.parse(DirectConfig), DirectConfig())

    def test_every_field_is_settable(self):
        config = self.parse(LiteV3Config, '--dim', '32', '--fit-span-s', '.3', '--prior-span-s', '.6',
                            '--no-error-estimate', '--no-presence')
        self.assertEqual((config.dim, config.fit_span_s, config.prior_span_s), (32, .3, .6))
        self.assertFalse(config.error_estimate)
        self.assertFalse(config.presence)
        self.assertEqual(self.parse(DirectConfig, '--fusion-blocks', '3').fusion_blocks, 3)

    def test_invalid_config_is_rejected(self):
        for overrides in (dict(dim=10, heads=4), dict(event_span_s=0.), dict(fit_span_s=0.)):
            with self.subTest(**overrides), self.assertRaises(ValueError):
                LiteV3Config(**overrides)


if __name__ == '__main__':
    unittest.main()
