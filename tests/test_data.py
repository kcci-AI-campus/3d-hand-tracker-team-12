"""hand_tracking.data: event recovery from exports, windows, manifest checks, rig frame, hands in view."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from model_helpers import sim_clip
import torch
from gigahands_sim import Config, demo_motion, simulate, save_dataset
from hand_tracking.contracts import MODEL_INPUT_KEYS, TARGET_KEYS, WINDOW_SAMPLE_KEYS
from hand_tracking.data import HandWindows, cached_clip, make_sample, read_clip, read_manifest


def synthetic_clip(events=6, queries=4):
    return dict(window_starts=np.array([0]), window=queries, query_time=np.arange(queries)*.05,
                event_arrival=np.arange(events)*.03, event_capture=np.arange(events)*.03-.01,
                event_features=np.ones((events,2,21,14), np.float32), event_valid=np.ones((events,2,21), bool),
                event_camera=np.arange(events) % 3, target_xyz=np.zeros((queries,2,21,3), np.float32),
                target_mask=np.ones((queries,2,21), bool), calibration_target=np.full((3,6), np.nan, np.float32),
                world_unit_cm=40.)


class WindowTests(unittest.TestCase):
    def test_test_split_is_disjoint_and_not_augmented(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);rows=[]
            for i,split in enumerate(('train','val','test')):
                (root/(split+'.npz')).touch()
                rows.append(dict(file=split+'.npz',split=split,participant=f'p{i}',source=split,windows=1))
            path=root/'manifest.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in rows))
            self.assertEqual(len(read_manifest(root)),3)
            rows[-1]['participant']='p0'
            path.write_text('\n'.join(json.dumps(r) for r in rows))
            with self.assertRaises(ValueError):read_manifest(root)

    def test_last_queries_only(self):
        clip = synthetic_clip()
        full, last = make_sample(clip, 0, max_events=8), make_sample(clip, 0, max_events=8, queries=2)
        self.assertEqual(len(last['query_times']), 2)
        torch.testing.assert_close(last['query_times'], full['query_times'][-2:])
        torch.testing.assert_close(last['target'], full['target'][-2:])
        self.assertLessEqual(int(last['event_present'].sum()), int(full['event_present'].sum()))
        with self.assertRaises(ValueError):
            make_sample(clip, 0, queries=clip['window']+1)

    def test_hands_out_of_every_view_have_no_position_target(self):
        clip = synthetic_clip(events=6, queries=4)
        clip['hand_in_view'] = np.array([[True, True], [True, False], [True, False], [True, True]])
        masked = make_sample(clip, 0, max_events=8)
        kept = make_sample(clip, 0, max_events=8, mask_out_of_view=False)
        np.testing.assert_array_equal(masked['hand_in_view'].numpy(), clip['hand_in_view'])
        np.testing.assert_array_equal(masked['target_mask'].any(-1).numpy(), clip['hand_in_view'])
        self.assertTrue(kept['target_mask'].all())
        # Without the simulator's cameras every hand counts as in view.
        self.assertTrue(make_sample(synthetic_clip(), 0, max_events=8)['hand_in_view'].all())

    def test_make_sample_matches_window_contract(self):
        sample = make_sample(synthetic_clip(), 0, max_events=8)
        self.assertEqual(tuple(sample), WINDOW_SAMPLE_KEYS)
        self.assertEqual(set(sample), set(MODEL_INPUT_KEYS+TARGET_KEYS))
        self.assertEqual(int(sample['event_present'].sum()), 6)

    def test_events_recovered_from_export_and_windows(self):
        positions, times = demo_motion(90)
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
            np.testing.assert_array_equal(sample['target_world'].numpy(), clip['target_xyz'][start:start+window])
            np.testing.assert_array_equal(sample['target'].numpy(), clip['rig_target'][start:start+window])

            rows = [dict(file=f'{s}.npz', split=s, participant=p, source=s, windows=len(data["window_starts"]))
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

    def test_cached_clip_matches_npz_and_follows_its_source(self):
        positions, times = demo_motion(60)
        with tempfile.TemporaryDirectory() as folder:
            root, cache = Path(folder)/'data', Path(folder)/'cache'
            root.mkdir()
            save_dataset(root/'a.npz', simulate(positions, times, Config()))
            direct = cached_clip(root, 'a.npz')
            written, loaded = cached_clip(root, 'a.npz', cache), cached_clip(root, 'a.npz', cache)
            self.assertTrue((cache/'a.npz').is_file())
            arrays = set(direct)                          # before make_sample memoises '_causal' in a clip
            for clip in (written, loaded):
                self.assertEqual(set(clip)-{'_causal'}, arrays)
                for key in arrays:
                    np.testing.assert_array_equal(clip[key], direct[key])
                sample, expected = make_sample(clip, 2, max_events=64), make_sample(direct, 2, max_events=64)
                for key in expected:
                    torch.testing.assert_close(sample[key], expected[key])
            # A changed source rebuilds the entry.
            save_dataset(root/'a.npz', simulate(positions[:40], times[:40], Config()))
            np.testing.assert_array_equal(cached_clip(root, 'a.npz', cache)['window_starts'],
                                          read_clip(root/'a.npz')['window_starts'])


class CalibrationTargetTests(unittest.TestCase):
    def test_rig_frame_removes_only_the_rig_wide_error(self):
        """The rig frame is the world moved by one similarity per clip: hands keep their shape,
        the calibration target shrinks to the cameras' error relative to each other, and the
        corrected rays still triangulate the rig-frame target."""
        with tempfile.TemporaryDirectory() as folder:
            clip = sim_clip(folder, fit_extent=.25, hand_dropout_prob=0, burst_dropout_prob=0, hand_count_timing=False,
                            correlated_pixel_std=0, randomize_errors=False, position_std=.1, angle_std_deg=5.,
                            focal_std_pct=0., principal_std_px=0., k1=0., k2=0., pixel_std=0.)
        world, rig = make_sample(clip, 4, target_frame='world'), make_sample(clip, 4)
        valid = world['target_mask']
        # Same hand shape: pairwise joint distances agree up to the similarity's scale.
        pairs = lambda x: torch.cdist(x[valid][None].double(), x[valid][None].double())[0]
        scale = float(pairs(rig['target']).sum()/pairs(world['target']).sum())
        self.assertLess(abs(scale-1), .1)
        torch.testing.assert_close(pairs(rig['target']), pairs(world['target'])*scale, atol=1e-4, rtol=0)
        torch.testing.assert_close(rig['target_world'], world['target'])
        norms = lambda sample: (sample['calibration_target']/torch.tensor([5*torch.pi/180]*3+[.1]*3)).square().sum()
        self.assertLess(float(norms(rig)), float(norms(world)))

    def test_simulated_hands_are_in_view(self):
        with tempfile.TemporaryDirectory() as folder:
            clip = sim_clip(folder)
        self.assertEqual(clip['hand_in_view'].shape, (len(clip['target_xyz']), 2))
        self.assertGreater(clip['hand_in_view'].mean(), .9)                       # the demo motion stays in view


if __name__ == '__main__':
    unittest.main()
