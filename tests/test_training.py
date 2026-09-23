"""hand_tracking.engine, checkpoints and config: metrics, checkpoint round trips, CLI generation."""
import argparse
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from model_helpers import small
import torch
from hand_tracking.checkpoints import load_checkpoint, restore_training, save_checkpoint, training_checkpoint
from hand_tracking.config import ModelConfig, SamplingConfig, add_model_arguments, model_config_from_args
from hand_tracking.contracts import ModelOutput
from hand_tracking.engine import run_epoch
from hand_tracking.events import accepted_captures
from hand_tracking.model import HandTransformer

CPU = torch.device('cpu')


class ZeroModel(torch.nn.Module):
    """Predicts zero pose and calibration, so metrics reduce to the targets."""
    config = ModelConfig()

    def forward(self, features, valid, camera, capture, arrival, present, query, return_details=False):
        b, e = present.shape
        pose = torch.zeros(b, query.shape[1], 2, 21, 3)
        return ModelOutput(pose, torch.zeros(b, e, 3, 6), accepted_captures(camera, capture, present))


def calibration_batch(target, events=2):
    b = len(target)
    return dict(event_features=torch.zeros(b,events,2,21,14), event_valid=torch.ones(b,events,2,21, dtype=torch.bool),
                event_camera=torch.zeros(b,events, dtype=torch.long),
                event_capture=(torch.arange(events).float()*.01-.1).expand(b,-1),
                event_arrival=(torch.arange(events).float()*.01-.09).expand(b,-1),
                event_present=torch.ones(b,events, dtype=torch.bool), query_times=torch.zeros(b,2),
                target=torch.zeros(b,2,2,21,3), target_mask=torch.ones(b,2,2,21, dtype=torch.bool),
                world_unit_cm=torch.full((b,), 30.), calibration_target=target)


class MetricTests(unittest.TestCase):
    def test_partial_calibration_metric_is_camera_event_weighted(self):
        scale = torch.tensor([5*torch.pi/180]*3+[.1]*3)
        targets = torch.full((2,3,6), float('nan'))
        targets[0,0] = scale          # 1 known camera, squared error 1
        targets[1] = scale*2          # 3 known cameras, squared error 4
        model = ZeroModel()
        mse = lambda *batches: run_epoch(model, list(batches), CPU)['calibration_mse']
        self.assertAlmostEqual(mse(calibration_batch(targets[:1])), 1., places=6)
        self.assertAlmostEqual(mse(calibration_batch(targets)), 3.25, places=6)
        self.assertAlmostEqual(mse(calibration_batch(targets[:1]), calibration_batch(targets[1:])), 3.25, places=6)
        self.assertAlmostEqual(mse(calibration_batch(targets[:1],1), calibration_batch(targets[1:],3)), 3.7, places=6)
        duplicate = calibration_batch(targets[1:], 3)
        duplicate['event_capture'] = torch.full((1,3), -.1)                    # only the first is accepted
        self.assertAlmostEqual(mse(calibration_batch(targets[:1],1), duplicate), 3.25, places=6)
        padded = calibration_batch(targets)
        padded['event_present'][:,1] = False                                  # padding carries no target
        self.assertAlmostEqual(mse(padded), 3.25, places=6)
        self.assertIsNone(mse(calibration_batch(torch.full_like(targets, float('nan')))))

    def test_final_query_metrics(self):
        batch = calibration_batch(torch.full((1,3,6), float('nan')))
        batch['target'][:,-1,...,0] = .1                                      # 3 cm at 30 cm per unit
        report = run_epoch(ZeroModel(), [batch], CPU)
        self.assertAlmostEqual(report['mpjpe_mm'], 30., places=4)
        self.assertEqual((report['pck20'], report['samples'], report['joints'], report['batches']), (0., 1, 42, 1))


class CheckpointTests(unittest.TestCase):
    def test_legacy_checkpoint_restores_input_layout_and_weights(self):
        model = HandTransformer(small()).eval()
        saved = dict(model=model.state_dict(), model_config=asdict(model.config), training_options=dict(max_events=37))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'legacy.pt'
            torch.save(saved, path)
            restored = load_checkpoint(path)
        self.assertEqual(restored.sampling.max_events, 37)
        self.assertFalse(restored.model.training)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(restored.model.state_dict()[name], value, atol=0, rtol=0)

    def test_explicit_sampling_config_takes_precedence_with_legacy_fallback(self):
        self.assertEqual(SamplingConfig.from_checkpoint({}).max_events, 128)
        saved = dict(sampling_config=dict(max_events=48), training_options=dict(max_events=37))
        self.assertEqual(SamplingConfig.from_checkpoint(saved).max_events, 48)

    def test_training_state_round_trip(self):
        def objects(seed):
            torch.manual_seed(seed)
            model = HandTransformer(small())
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
    def parse(self, *argv):
        parser = argparse.ArgumentParser()
        add_model_arguments(parser)
        return model_config_from_args(parser.parse_args(list(argv)))

    def test_defaults_round_trip(self):
        self.assertEqual(self.parse(), ModelConfig())

    def test_every_field_is_settable(self):
        config = self.parse('--dim', '64', '--anchor-hold-s', '.25', '--sample-max-age-s', '.15',
                            '--anchor-ray-gate', '.2', '--anchor-motion-fit', '--no-calibration-head')
        self.assertEqual((config.dim, config.anchor_hold_s, config.sample_max_age_s, config.anchor_ray_gate),
                         (64, .25, .15, .2))
        self.assertTrue(config.anchor_motion_fit)
        self.assertFalse(config.calibration_head)

    def test_invalid_config_is_rejected(self):
        for overrides in (dict(dim=10, heads=4), dict(event_span_s=0.), dict(anchor_ray_gate=0.)):
            with self.subTest(**overrides), self.assertRaises(ValueError):
                ModelConfig(**overrides)


if __name__ == '__main__':
    unittest.main()
