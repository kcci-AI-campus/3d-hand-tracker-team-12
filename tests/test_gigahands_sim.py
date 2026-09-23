import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from gigahands_sim import Config, demo_motion, load_motion, simulate, training_window, save_dataset, bounded_normal_vector, load_timing_profile, replay_camera_timing, look_at, camera_pose_without_roll


class SimulationTests(unittest.TestCase):
    def test_camera_roll_is_zero_with_random_yaw_pitch(self):
        rng = np.random.default_rng(123)
        for target in ([0.,0.,0.], [1.,2.,3.], [-1.,10.,.5]):
            nominal = look_at(np.array([-1.,-1.,.5]), target)
            samples = []
            for _ in range(200):
                actual, offsets = camera_pose_without_roll(rng, nominal, 5)
                np.testing.assert_allclose(actual @ actual.T, np.eye(3), atol=1e-12)
                self.assertAlmostEqual(np.linalg.det(actual), 1)
                self.assertEqual(actual[0,1], 0)
                self.assertLessEqual(actual[1,1], 0)
                samples.append(offsets)
            self.assertTrue(np.all(np.std(samples,axis=0)>4))
        p,t = demo_motion(30)
        data = simulate(p,t,Config())
        np.testing.assert_allclose(data['actual_roll_deg'],0,atol=1e-12)
        self.assertEqual(data['sampled_yaw_pitch_errors_deg'].shape,(3,2))
        self.assertFalse(json.loads(data['metadata'])['units']['camera_roll_enabled'])

    def test_pose_seed_changes_only_installation_randomness(self):
        p, t = demo_motion(60)
        a = simulate(p, t, Config(pose_seed=11))
        b = simulate(p, t, Config(pose_seed=12))
        repeat = simulate(p, t, Config(pose_seed=12))
        self.assertFalse(np.array_equal(a['actual_origins'], b['actual_origins']))
        self.assertFalse(np.array_equal(a['actual_rotations'], b['actual_rotations']))
        for name in ('target_xyz', 'arrival_time', 'physical_capture_time', 'selected_frame', 'actual_intrinsics'):
            np.testing.assert_array_equal(a[name], b[name])
        np.testing.assert_array_equal(b['features'], repeat['features'])

    def test_unbounded_installation_errors(self):
        cfg = Config()
        cfg.validate()
        self.assertIsNone(cfg.position_limit_cm)
        self.assertIsNone(cfg.angle_limit_deg)
        self.assertEqual(cfg.error_ranges['position_std'], [.1, .1])
        self.assertEqual(cfg.error_ranges['angle_std_deg'], [5, 5])
        rng = np.random.default_rng(123)
        samples = np.array([bounded_normal_vector(rng, 3, None) for _ in range(10000)])
        np.testing.assert_allclose(samples.std(axis=0), 3, atol=.08)
        self.assertGreater(np.linalg.norm(samples, axis=1).max(), 10)
        p, t = demo_motion(30)
        data = simulate(p, t, cfg)
        meta = json.loads(data['metadata'])
        self.assertIsNone(meta['config']['position_limit_cm'])
        self.assertEqual(meta['config']['angle_std_deg'], 5)

    def clean(self):
        return Config(timing_profile='', randomize_errors=False, position_std=0, angle_std_deg=0, focal_std_pct=0, principal_std_px=0,
                      pixel_std=0, outlier_prob=0, missing_prob=0, packet_loss=0,
                      latency_ms=0, latency_jitter_ms=0, processing_ms=0, capture_jitter_ms=0)

    def test_exact_rays_static_geometry(self):
        p, t = demo_motion(60); p[:] = p[0]
        d = simulate(p, t, self.clean())
        for c in range(3):
            delta = d['target_xyz'] - d['nominal_origins'][c]
            expected = delta / np.linalg.norm(delta, axis=-1, keepdims=True)
            mask = d['input_mask'][:, c]
            self.assertGreater(mask.sum(), 0)
            np.testing.assert_allclose(d['ray_directions'][:,c][mask], expected[mask], atol=2e-7)

    def test_causal_reproducible_and_reordering(self):
        p, t = demo_motion(120)
        cfg = Config(timing_profile='', randomize_errors=False, latency_ms=80, latency_jitter_ms=100, packet_loss=.2)
        d = simulate(p, t, cfg); other = simulate(p, t, cfg)
        np.testing.assert_array_equal(d['features'], other['features'])
        for c in range(3):
            valid = d['selected_frame'][:,c] >= 0
            self.assertTrue(np.all(d['arrival_time'][valid,c] <= d['query_time'][valid]))
            self.assertTrue(np.all(d['physical_capture_time'][valid,c] <= d['arrival_time'][valid,c]))
            self.assertTrue(np.all(np.diff(d['selected_frame'][valid,c]) >= 0))
        x, m, y, ym = training_window(d, 5)
        self.assertEqual(x.shape, (cfg.window*126,14))
        self.assertTrue(np.all(x[m,9] <= 1e-7))
        self.assertTrue(np.all(x[~m] == 0))
        self.assertEqual(y.shape, (2,21,3))

    def test_packet_loss_and_missing_targets(self):
        p,t = demo_motion(40); p[:,0,5] = np.nan
        d = simulate(p,t,Config(randomize_errors=False, packet_loss=1))
        self.assertFalse(d['input_mask'].any())
        self.assertTrue(np.isfinite(d['features']).all())
        self.assertFalse(d['target_mask'][:,0,5].any())
        self.assertTrue(np.all(d['features'] == 0))

    def test_loader_and_roundtrip(self):
        p,t = demo_motion(20)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'motion.json'
            path.write_text(json.dumps(p.reshape(20,126).tolist()))
            loaded, times = load_motion(path)
            np.testing.assert_allclose(loaded,p); np.testing.assert_allclose(times,t)
            d = simulate(loaded,times,Config(window=4, intrinsics=[[400,400,160,120]]*3))
            out = Path(folder)/'out.npz'; save_dataset(out,d,path)
            with np.load(out,allow_pickle=False) as saved:
                x,mask,y,ym = training_window(saved,0)
                self.assertTrue(np.isfinite(x).all())
                self.assertEqual(json.loads(str(saved['metadata']))['schema_version'],1)

    def test_invalid_settings_and_workspace(self):
        p,t = demo_motion(20)
        for cfg in [Config(camera_fps=0),Config(window=1.5),Config(packet_loss=2),Config(axis_order='xxx'),Config(cameras=[[0,0,0]]*3),Config(scale=100,fit_extent=0)]:
            with self.assertRaises((ValueError,TypeError)): simulate(p,t,cfg)

    def test_rolling_shutter_and_clock(self):
        p,t = demo_motion(80)
        d = simulate(p,t,Config(timing_profile='', randomize_errors=False, rolling_shutter_ms=20, clock_offset_std_ms=30, clock_drift_std_ppm=100))
        self.assertTrue(np.isfinite(d['features']).all())
        valid = d['selected_frame'] >= 0
        self.assertTrue(np.all((d['arrival_time']-d['physical_capture_time'])[valid] >= .03-1e-9))

    def test_random_scenarios_keep_optics_and_reproduce(self):
        p,t = demo_motion(50)
        cfg = Config()
        a = simulate(p,t,cfg); b = simulate(p,t,cfg); c = simulate(p,t,Config(seed=43))
        np.testing.assert_array_equal(a['features'], b['features'])
        np.testing.assert_array_equal(a['actual_intrinsics'], a['nominal_intrinsics'])
        np.testing.assert_array_equal(a['actual_intrinsics'], c['actual_intrinsics'])
        self.assertFalse(np.array_equal(a['features'], c['features']))
        meta = json.loads(a['metadata'])
        for name,value in meta['randomization']['sampled_parameters'].items():
            lo,hi = cfg.error_ranges[name]
            self.assertTrue(lo <= value <= hi)
            self.assertEqual(meta['config'][name], value)
        with self.assertRaises(ValueError): simulate(p,t,Config(error_ranges={'width':[100,400]}))

    def test_physical_units_and_pose_limits(self):
        p,t = demo_motion(30)
        d = simulate(p,t,Config(randomize_errors=False, position_std=10, angle_std_deg=100, position_limit_cm=3, angle_limit_deg=10))
        self.assertEqual(json.loads(d['metadata'])['units']['world_unit_cm'],30)
        np.testing.assert_allclose(d['nominal_origins'] * 30, [[-30,-30,15],[30,-30,15],[0,30,30]])
        self.assertTrue(np.all(d['actual_position_errors_cm'] <= 3+1e-10))
        self.assertTrue(np.all(d['actual_angle_errors_deg'] <= 10+1e-10))
        zero = simulate(p,t,Config(position_limit_cm=0,angle_limit_deg=0))
        np.testing.assert_allclose(zero['actual_origins'],zero['nominal_origins'])
        np.testing.assert_allclose(zero['actual_rotations'],zero['nominal_rotations'])
        rng = np.random.default_rng(11)
        for sigma in (.03, .1, 10):
            samples = np.array([bounded_normal_vector(rng,sigma,.1) for _ in range(1000)])
            self.assertTrue(np.all(np.linalg.norm(samples,axis=1) <= .1))

    def test_empirical_timing_pairs_and_no_double_delay(self):
        observations, metadata = load_timing_profile(Config().timing_profile)
        ct, delay, indices = replay_camera_timing(observations, 400, np.random.default_rng(8))
        np.testing.assert_allclose(np.diff(ct)*1000, observations[indices[1:],0], atol=1e-7)
        np.testing.assert_allclose(delay*1000, observations[indices,1:].sum(1))
        p,t = demo_motion(60)
        cfg = Config(randomize_errors=False, processing_ms=9999, latency_ms=9999, rolling_shutter_ms=9999,
                     missing_prob=0, packet_loss=0)
        d=simulate(p,t,cfg)
        present=d['selected_frame']>=0
        actual=(d['arrival_time']-d['physical_capture_time'])[present]*1000
        self.assertTrue(np.all(actual <= observations[:,1:].sum(1).max()+1e-6))
        np.testing.assert_array_equal(d['input_mask'],d['in_frame_mask'])
        self.assertEqual(json.loads(d['metadata'])['timing_replay']['rows'],5015)


if __name__ == '__main__': unittest.main()
