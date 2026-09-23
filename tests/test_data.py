"""hand_tracking.data: event recovery from exports, windows, manifest checks, rig augmentation."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from model_helpers import batch_of, sim_clip
import torch
from gigahands_sim import Config, demo_motion, simulate, save_dataset
from hand_tracking.contracts import MODEL_INPUT_KEYS, TARGET_KEYS, WINDOW_SAMPLE_KEYS
from hand_tracking.data import HandWindows, make_sample, random_rigid, read_clip, read_manifest
from hand_tracking.events import event_anchor
from hand_tracking.geometry import apply_calibration


def synthetic_clip(events=6, queries=4):
    return dict(window_starts=np.array([0]), window=queries, query_time=np.arange(queries)*.05,
                event_arrival=np.arange(events)*.03, event_capture=np.arange(events)*.03-.01,
                event_features=np.ones((events,2,21,14), np.float32), event_valid=np.ones((events,2,21), bool),
                event_camera=np.arange(events) % 3, target_xyz=np.zeros((queries,2,21,3), np.float32),
                target_mask=np.ones((queries,2,21), bool), calibration_target=np.full((3,6), np.nan, np.float32),
                world_unit_cm=30.)


class WindowTests(unittest.TestCase):
    def test_make_sample_matches_window_contract(self):
        sample = make_sample(synthetic_clip(), 0, max_events=8)
        self.assertEqual(tuple(sample), WINDOW_SAMPLE_KEYS)
        self.assertEqual(set(sample), set(MODEL_INPUT_KEYS+TARGET_KEYS))
        self.assertEqual(int(sample['event_present'].sum()), 6)

    def test_events_recovered_from_export_and_windows(self):
        positions, times = demo_motion(30)
        data = simulate(positions, times, Config())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            save_dataset(root/'train.npz', data)
            save_dataset(root/'val.npz', data)
            clip = read_clip(root/'train.npz')
            # Every frame the export selects is one event (with or without detections), at its arrival time.
            arrival = np.asarray(data['arrival_time'])
            selected = np.asarray(data['selected_frame'])
            shown = sorted({(c, int(selected[i,c]), round(float(arrival[i,c]), 5))
                            for i in range(len(arrival)) for c in range(3) if selected[i,c] >= 0})
            shown = [(c, a) for c, _, a in shown]
            recovered = sorted((int(c), round(float(a), 5)) for c, a in zip(clip['event_camera'], clip['event_arrival']))
            self.assertEqual(recovered, shown)
            self.assertTrue((np.diff(clip['event_arrival']) >= 0).all())
            self.assertNotIn('actual_origins', clip)

            sample = make_sample(clip, 4, context_s=.5, max_events=64)
            start, window = int(clip['window_starts'][4]), clip['window']
            last = clip['query_time'][start+window-1]
            present = sample['event_present'].numpy()
            np.testing.assert_allclose(sample['query_times'].numpy(), clip['query_time'][start:start+window]-last,
                                       atol=1e-6)
            self.assertTrue((sample['event_arrival'].numpy()[present] <= 1e-6).all())
            self.assertTrue((sample['event_capture'].numpy()[present] >= clip['query_time'][start]-last-.5-1e-6).all())
            np.testing.assert_array_equal(sample['target'].numpy(), clip['target_xyz'][start:start+window])

            rows = [dict(file=f'{s}.npz', split=s, participant=p, source=s, windows=15)
                    for s, p in [('train','p001'), ('val','p002')]]
            manifest = root/'manifest.jsonl'
            manifest.write_text('\n'.join(json.dumps(r) for r in rows))
            dataset = HandWindows(root, 'val', windows_per_clip=3, context_s=.5, max_events=64)
            first, second = list(dataset), list(dataset)
            self.assertEqual(len(first), 3)
            self.assertEqual(first[0]['event_features'].shape[0], 64)
            torch.testing.assert_close(first[1]['event_features'], second[1]['event_features'])
            rows[1]['participant'] = 'p001'
            manifest.write_text('\n'.join(json.dumps(r) for r in rows))
            with self.assertRaises(ValueError):
                read_manifest(root)


class AugmentationTests(unittest.TestCase):
    def test_rig_augmentation_is_rigid(self):
        with tempfile.TemporaryDirectory() as folder:
            clip = sim_clip(folder)
        rotation, shift = random_rigid(np.random.default_rng(0), 45, .25)
        plain, moved = make_sample(clip, 4), make_sample(clip, 4, (rotation, shift))
        r = torch.tensor(rotation, dtype=torch.float32)
        s = torch.tensor(shift, dtype=torch.float32)
        valid = plain['target_mask']
        torch.testing.assert_close(moved['target'][valid], plain['target'][valid]@r.T+s, atol=1e-5, rtol=0)
        a, b = event_anchor(*batch_of(plain)), event_anchor(*batch_of(moved))
        torch.testing.assert_close(b, a@r.T+s, atol=1e-3, rtol=0)

    def test_calibration_target_matches_simulator_rig(self):
        with tempfile.TemporaryDirectory() as folder:
            clip = sim_clip(folder, randomize_errors=False, position_std=.1, angle_std_deg=5., focal_std_pct=0.,
                            principal_std_px=0., k1=0., k2=0., pixel_std=0.)
        rotation, shift = random_rigid(np.random.default_rng(1), 45, .25)
        for rigid in (None, (rotation, shift)):
            sample = make_sample(clip, 4, rigid)
            target = sample['calibration_target'][None]
            features = sample['event_features'][None].clone()
            per_camera = lambda channels: features[:,:,None,...,channels].expand(-1,-1,3,-1,-1,-1)
            o, d = apply_calibration(target, per_camera(slice(2,5)), per_camera(slice(5,8)))
            own = sample['event_camera'][None,:,None,None,None,None].expand(-1,-1,1,2,21,3)
            fixed = features.clone()
            fixed[...,2:5] = o.gather(2, own)[:,:,0]
            fixed[...,5:8] = d.gather(2, own)[:,:,0]
            inputs = batch_of(sample)
            error = lambda f: (event_anchor(f, *inputs[1:])[0,-1]-sample['target'][-1]).norm(dim=-1)[
                sample['target_mask'][-1]].mean().item()
            # The true correction removes the calibration error (up to camera timing/motion).
            self.assertLess(error(fixed), error(features)*.3)


if __name__ == '__main__':
    unittest.main()
