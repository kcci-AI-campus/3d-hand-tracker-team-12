"""hand_tracking.data: event recovery from exports, windows, manifest checks, rig augmentation."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from model_helpers import batch_of, sim_clip, lite_anchor
import torch
from gigahands_sim import Config, demo_motion, simulate, save_dataset
from hand_tracking.contracts import MODEL_INPUT_KEYS, TARGET_KEYS, WINDOW_SAMPLE_KEYS
from hand_tracking.data import HandWindows, cached_clip, make_sample, read_clip, read_manifest
from hand_tracking.geometry import apply_calibration


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

    def test_hands_the_input_never_shows_have_no_target(self):
        clip = synthetic_clip(events=6, queries=4)    # events every 30 ms, queries every 50 ms
        clip['event_valid'][:, 1] = False              # hand 1 never detected
        clip['event_valid'][:2, 0] = False             # hand 0 only from the third event (60 ms) on
        kept = make_sample(clip, 0, max_events=8)['target_mask']
        seen = make_sample(clip, 0, max_events=8, seen_span_s=.5)['target_mask']
        self.assertTrue(kept.all())
        self.assertFalse(seen[:, 1].any())
        # Hand 0's first detection arrives at 60 ms: queries at 0 and 50 ms have not seen it yet.
        self.assertEqual(seen[:, 0].all(-1).tolist(), [False, False, True, True])
        # At 100 ms a 10 ms span only reaches back to captures from 90 ms: none has arrived yet.
        self.assertFalse(make_sample(clip, 0, max_events=8, seen_span_s=.01)['target_mask'][2].any())

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
            for clip in (written, loaded):
                self.assertEqual(set(clip), set(direct))
                for key, value in direct.items():
                    np.testing.assert_array_equal(clip[key], value)
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
        from hand_tracking.geometry import apply_calibration
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
        # Each frame's corrections on the nominal rays reproduce that frame's target equally
        # well (the remainder is camera timing and motion).
        def corrected_error(sample):
            features = sample['event_features'][None].clone()
            per_camera = lambda channels: features[:,:,None,...,channels].expand(-1,-1,3,-1,-1,-1)
            o, d = apply_calibration(sample['calibration_target'][None], per_camera(slice(2,5)), per_camera(slice(5,8)))
            own = sample['event_camera'][None,:,None,None,None,None].expand(-1,-1,1,2,21,3)
            features[...,2:5], features[...,5:8] = o.gather(2, own)[:,:,0], d.gather(2, own)[:,:,0]
            error = (lite_anchor(features, *batch_of(sample)[1:])[0,-1]-sample['target'][-1]).norm(dim=-1)
            return float(error[sample['target_mask'][-1]].mean())
        self.assertLess(abs(corrected_error(rig)-corrected_error(world)), .05*corrected_error(world))

    def test_calibration_target_matches_simulator_rig(self):
        with tempfile.TemporaryDirectory() as folder:
            clip = sim_clip(folder, fit_extent=.25, hand_dropout_prob=0, burst_dropout_prob=0, hand_count_timing=False, correlated_pixel_std=0, randomize_errors=False, position_std=.1, angle_std_deg=5., focal_std_pct=0.,
                            principal_std_px=0., k1=0., k2=0., pixel_std=0.)
        sample = make_sample(clip, 4)
        target = sample['calibration_target'][None]
        features = sample['event_features'][None].clone()
        per_camera = lambda channels: features[:,:,None,...,channels].expand(-1,-1,3,-1,-1,-1)
        o, d = apply_calibration(target, per_camera(slice(2,5)), per_camera(slice(5,8)))
        own = sample['event_camera'][None,:,None,None,None,None].expand(-1,-1,1,2,21,3)
        fixed = features.clone()
        fixed[...,2:5] = o.gather(2, own)[:,:,0]
        fixed[...,5:8] = d.gather(2, own)[:,:,0]
        inputs = batch_of(sample)
        error = lambda f: (lite_anchor(f, *inputs[1:])[0,-1]-sample['target'][-1]).norm(dim=-1)[
            sample['target_mask'][-1]].mean().item()
        # The true correction removes the calibration error (up to camera timing/motion).
        self.assertLess(error(fixed), error(features)*.3)


if __name__ == '__main__':
    unittest.main()
