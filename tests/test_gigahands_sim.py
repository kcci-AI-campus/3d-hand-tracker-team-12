import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from gigahands_sim import Config, demo_motion, load_motion, simulate, training_window, save_dataset, bounded_normal_vector, load_timing_profile, replay_camera_timing, look_at, camera_pose_without_roll


class SimulationTests(unittest.TestCase):
    def test_camera3_local_ready_triggers_causal_queries(self):
        p,t=demo_motion(180)
        cfg=Config(randomize_errors=False,timing_profile='',fit_extent=.25,
                   position_std=0,angle_std_deg=0,packet_loss=0,
                   inference_stall_prob=.3,inference_jitter_ms=3)
        self.assertTrue(cfg.query_sync_camera3)
        data=simulate(p,t,cfg)
        events=data['hand_timing_events']
        arrivals=events[events[:,0]==2,2]
        expected=np.unique(arrivals[arrivals<=t[-1]])
        np.testing.assert_array_equal(data['query_time'],expected)
        self.assertGreater(np.std(np.diff(expected)),.001)
        used=data['selected_frame']>=0
        self.assertTrue((data['arrival_time'][used]<=np.broadcast_to(expected[:,None],used.shape)[used]).all())
        self.assertTrue((data['selected_frame'][:,2]>=0).all())
        cfg.packet_loss=.3
        lost=simulate(p,t,cfg)
        np.testing.assert_array_equal(lost['query_time'],expected)
        cfg.packet_loss=1
        local=simulate(p,t,cfg)
        np.testing.assert_array_equal(local['query_time'],expected)
        self.assertTrue((local['selected_frame'][:,:2]==-1).all())
        self.assertTrue((local['selected_frame'][:,2]>=0).all())
        rows=events[events[:,0]==2]
        np.testing.assert_allclose(rows[:,2]-rows[:,1],rows[:,4]/1000,atol=1e-12)
        np.testing.assert_array_equal(local['capture_time'][:,2],local['physical_capture_time'][:,2])
        cfg.latency_ms=500
        cfg.clock_offset_std_ms=500
        cfg.clock_drift_std_ppm=500
        np.testing.assert_array_equal(simulate(p,t,cfg)['query_time'],expected)

    def test_camera3_sync_with_fixed_and_profile_capture_timing(self):
        p,t=demo_motion(120)
        for profile in ('',Config().timing_profile):
            cfg=Config(hand_count_timing=False,timing_profile=profile,
                       randomize_errors=False,packet_loss=0,latency_jitter_ms=0)
            data=simulate(p,t,cfg)
            np.testing.assert_array_equal(data['arrival_time'][:,2],data['query_time'])
            np.testing.assert_array_equal(data['query_time'],simulate(p,t,cfg)['query_time'])

    def test_40cm_defaults_preserve_metric_bones_before_augmentation(self):
        from gigahands_sim import normalize
        cfg=Config()
        self.assertEqual(cfg.world_unit_cm,40)
        self.assertEqual(cfg.fit_extent,0)
        self.assertTrue(cfg.randomize_hand_shape and cfg.randomize_errors and cfg.hand_count_timing)
        self.assertAlmostEqual(cfg.position_std*cfg.world_unit_cm,2)
        p,_=demo_motion(20)
        normalized,_=normalize(p,cfg)
        np.testing.assert_allclose(np.linalg.norm(normalized[:,:,9]-normalized[:,:,0],axis=-1)*40,
                                   np.linalg.norm(p[:,:,9]-p[:,:,0],axis=-1)*100)

    def test_whole_hand_screen_threshold_and_dropout(self):
        from gigahands_sim import hand_screen_visibility
        cfg=Config()
        uv=np.full((1,2,21,2),100.);depth=np.ones((1,2,21))
        uv[0,0,:4,0]=-1;uv[0,1,:5,0]=-1
        allowed,counts=hand_screen_visibility(uv,depth,cfg)
        np.testing.assert_array_equal(allowed,[[True,False]])
        np.testing.assert_array_equal(counts,[[4,5]])
        p,t=demo_motion(90)
        cfg=Config(false_positive_prob=0,hand_count_timing=True,hand_dropout_prob=0)
        baseline=simulate(p,t,cfg)
        cfg.hand_dropout_prob=1
        cfg.distance_dropout_enabled=False
        dropped=simulate(p,t,cfg)
        self.assertFalse(dropped['input_mask'].any())
        np.testing.assert_array_equal(dropped['features'],0)
        np.testing.assert_array_equal(dropped['target_xyz'],baseline['target_xyz'])
        np.testing.assert_array_equal(dropped['target_mask'],baseline['target_mask'])
        np.testing.assert_array_equal(dropped['hand_timing_events'],baseline['hand_timing_events'])
        cfg.hand_dropout_prob=.05
        cfg.distance_dropout_enabled=True
        a=simulate(p,t,cfg);b=simulate(p,t,cfg)
        np.testing.assert_array_equal(a['hand_dropout_mask'],b['hand_dropout_mask'])
        for c in range(3):
            for index in np.unique(a['selected_frame'][:,c]):
                if index<0:continue
                values=a['hand_dropout_mask'][a['selected_frame'][:,c]==index,c]
                self.assertTrue((values==values[0]).all())
        with self.assertRaises(ValueError):Config(hand_outside_keypoint_threshold=0).validate()
        with self.assertRaises(ValueError):Config(hand_dropout_prob=1.1).validate()

    def test_hand_dependent_timing_and_transport_only(self):
        from gigahands_sim import hand_dependent_timing, normalize
        p,t=demo_motion(60);p[:]=p[0]
        cfg=Config(inference_jitter_ms=0,inference_stall_prob=0,fit_extent=.3,hand_count_timing=True,randomize_errors=False,position_std=0,angle_std_deg=0,
                   focal_std_pct=0,principal_std_px=0,processing_ms=9999)
        data=simulate(p,t,cfg)
        events=data['hand_timing_events']
        self.assertGreater(len(events),0)
        np.testing.assert_allclose(events[:,4],np.array(cfg.hand_inference_ms)[events[:,3].astype(int)])
        observations,_=load_timing_profile(cfg.timing_profile)
        meta=json.loads(data['metadata'])
        for c in range(3):
            e=events[events[:,0]==c]
            np.testing.assert_allclose(np.diff(e[:,1]),np.maximum(1/19,e[:-1,4]/1000))
            offset=meta['timing_replay']['start_rows'][c]
            np.testing.assert_allclose((e[:,2]-e[:,1])*1000-e[:,4],
                                       0 if c==2 else observations[(offset+np.arange(len(e)))%len(observations),2],atol=1e-8)
        self.assertTrue(np.all(data['features'][...,9][data['input_mask']]<=0))
        points,_=normalize(p,cfg)
        for count in (0,1,2):
            example=points.copy();example[:,count:]=np.nan
            ct,cost,counts=hand_dependent_timing(example,t,data['actual_origins'][0],data['actual_rotations'][0],data['actual_intrinsics'][0],cfg,np.random.default_rng(1))
            np.testing.assert_array_equal(counts,np.full(len(ct),count))
            np.testing.assert_allclose(cost,cfg.hand_inference_ms[count])
            np.testing.assert_allclose(np.diff(ct),max(1/19,cfg.hand_inference_ms[count]/1000))

    def test_first_two_cameras_share_y_displacement(self):
        p,t = demo_motion(20)
        ys=[]
        for seed in range(6):
            for limit in (None, 3., 0.):
                cfg=Config(query_sync_camera3=False, seed=seed, position_limit_cm=limit)
                cfg.cameras[1][1] = -.8  # Synchronize errors, not absolute coordinates.
                data=simulate(p,t,cfg)
                delta=data['actual_origins']-data['nominal_origins']
                self.assertAlmostEqual(delta[0,1],delta[1,1],places=14)
                if limit is None:
                    ys.append(delta[0,1])
                    self.assertFalse(np.allclose(delta[0,[0,2]],delta[1,[0,2]]))
                    self.assertNotAlmostEqual(delta[2,1],delta[0,1])
                    self.assertFalse(np.allclose(data['sampled_yaw_pitch_errors_deg'][0],data['sampled_yaw_pitch_errors_deg'][1]))
                else:
                    self.assertTrue(np.all(np.linalg.norm(delta,axis=1)*cfg.world_unit_cm<=limit+1e-10))
        self.assertGreater(np.std(ys),0)

    def test_hand_shape_bone_geometry_and_projection(self):
        from gigahands_sim import augment_hand_shape, normalize
        p,t = demo_motion(30)
        cfg = Config(query_sync_camera3=False, randomize_hand_shape=True, hand_count_timing=False)
        source,_ = normalize(p,cfg)
        changed,meta = augment_hand_shape(source,cfg)
        np.testing.assert_array_equal(changed[:,:,0],source[:,:,0])
        for finger in range(5):
            base=1+4*finger
            for joint in range(base,base+4):
                parent=0 if joint==base else joint-1
                factor=meta['size_factor']*(1 if joint==base else meta['finger_length_factors'][finger])
                np.testing.assert_allclose(changed[:,:,joint]-changed[:,:,parent],
                                           (source[:,:,joint]-source[:,:,parent])*factor,atol=1e-12)
        a=simulate(p,t,cfg); b=simulate(p,t,cfg)
        np.testing.assert_array_equal(a['features'],b['features'])
        np.testing.assert_allclose(a['target_xyz'],changed,atol=1e-7)
        plain=simulate(p,t,Config(query_sync_camera3=False, randomize_hand_shape=False, hand_count_timing=False))
        self.assertFalse(np.allclose(a['target_xyz'],plain['target_xyz']))
        np.testing.assert_array_equal(a['actual_origins'],plain['actual_origins'])
        np.testing.assert_array_equal(a['arrival_time'],plain['arrival_time'])
        self.assertFalse(np.allclose(a['pixels'],plain['pixels']))
        np.testing.assert_array_equal(augment_hand_shape(source,Config(randomize_hand_shape=False))[0],source)
        for value in (-1,100,float('nan')):
            with self.assertRaises(ValueError): Config(hand_size_percent=value).validate()

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
        a = simulate(p, t, Config(pose_seed=11, hand_count_timing=False))
        b = simulate(p, t, Config(pose_seed=12, hand_count_timing=False))
        repeat = simulate(p, t, Config(pose_seed=12, hand_count_timing=False))
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
        self.assertEqual(cfg.error_ranges['position_std'], [2/40, 2/40])
        self.assertEqual(cfg.error_ranges['angle_std_deg'], [3, 3])
        self.assertAlmostEqual(cfg.position_std*cfg.world_unit_cm, 2)
        rng = np.random.default_rng(123)
        samples = np.array([bounded_normal_vector(rng, 3, None) for _ in range(10000)])
        np.testing.assert_allclose(samples.std(axis=0), 3, atol=.08)
        self.assertGreater(np.linalg.norm(samples, axis=1).max(), 10)
        p, t = demo_motion(30)
        data = simulate(p, t, cfg)
        meta = json.loads(data['metadata'])
        self.assertIsNone(meta['config']['position_limit_cm'])
        self.assertEqual(meta['config']['angle_std_deg'], 3)

    def clean(self):
        return Config(false_positive_prob=0, burst_dropout_prob=0, correlated_pixel_std=0, randomize_hand_shape=False, hand_count_timing=False, hand_dropout_prob=0, timing_profile='', randomize_errors=False, position_std=0, angle_std_deg=0, focal_std_pct=0, principal_std_px=0,
                      pixel_std=0, outlier_prob=0, missing_prob=0, packet_loss=0,
                      latency_ms=0, latency_jitter_ms=0, processing_ms=0, capture_jitter_ms=0)

    def test_exact_rays_static_geometry(self):
        p, t = demo_motion(60); p[:] = p[0]
        cfg = self.clean(); cfg.fit_extent = .3  # All joints in view: test ray geometry, not hand rejection.
        d = simulate(p, t, cfg)
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
        d = simulate(p,t,Config(query_sync_camera3=False, randomize_errors=False, packet_loss=1))
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
                self.assertEqual(json.loads(str(saved['metadata']))['schema_version'],2)

    def test_invalid_settings_and_workspace(self):
        p,t = demo_motion(20)
        for cfg in [Config(camera_fps=0),Config(window=1.5),Config(packet_loss=2),Config(axis_order='xxx'),Config(cameras=[[0,0,0]]*3),Config(scale=100,fit_extent=0)]:
            with self.assertRaises((ValueError,TypeError)): simulate(p,t,cfg)

    def test_rolling_shutter_and_clock(self):
        p,t = demo_motion(80)
        d = simulate(p,t,Config(hand_count_timing=False, timing_profile='', randomize_errors=False, rolling_shutter_ms=20, clock_offset_std_ms=30, clock_drift_std_ppm=100))
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
        self.assertEqual(json.loads(d['metadata'])['units']['world_unit_cm'],40)
        np.testing.assert_allclose(d['nominal_origins'] * 40, [[-40,-40,20],[40,-40,20],[0,40,40]])
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
        cfg = Config(hand_count_timing=False, randomize_errors=False, processing_ms=9999, latency_ms=9999, rolling_shutter_ms=9999,
                     missing_prob=0, packet_loss=0)
        d=simulate(p,t,cfg)
        present=d['selected_frame']>=0
        actual=(d['arrival_time']-d['physical_capture_time'])[present]*1000
        self.assertTrue(np.all(actual <= observations[:,1:].sum(1).max()+1e-6))
        np.testing.assert_array_equal(d['input_mask'],(d['in_frame_mask'] & d['hand_detected_mask'][...,None]) | d['false_positive_mask'][...,None])
        self.assertEqual(json.loads(d['metadata'])['timing_replay']['rows'],5015)


if __name__ == '__main__': unittest.main()
