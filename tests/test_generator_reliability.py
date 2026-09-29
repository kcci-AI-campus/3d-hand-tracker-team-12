import io
import json
import tarfile
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
from gigahands_sim import Config, demo_motion, simulate, training_window, window_frame_offsets, hand_burst_mask, augment_hand_shape, normalize
from gigahands_balanced import run
from dataset_quality import clip_quality


class ReliabilityTests(unittest.TestCase):
    def test_overlap_coverage_depth_and_invalid_hands(self):
        from gigahands_sim import hand_overlap_probabilities
        cfg=Config()
        box=np.array([[20.,20.],[60.,20.],[60.,60.],[20.,60.]])
        uv=np.tile(np.resize(box,(21,2)),(6,2,1,1))
        depth=np.ones((6,2,21));depth[:,1]=2
        uv[1,1,:,0]+=100 # disjoint
        uv[2,1,:,0]+=20 # half overlap
        uv[3,1]=np.nan
        depth[4,1]=-1
        depth[5]=depth[0,::-1]
        score,prob=hand_overlap_probabilities(uv,depth,cfg)
        np.testing.assert_allclose(score[0],1)
        np.testing.assert_allclose(prob[0],[.2,.8])
        np.testing.assert_allclose(score[2],.5)
        self.assertTrue((prob[2]<prob[0]).all())
        np.testing.assert_array_equal(prob[[1,3,4]],0)
        np.testing.assert_allclose(prob[5],[.8,.2])
        np.testing.assert_array_equal(hand_overlap_probabilities(uv,depth,replace(cfg,overlap_dropout_max=0))[1],0)
        for bad in (-1,1.1,float('nan')):
            with self.assertRaises(ValueError):replace(cfg,overlap_dropout_max=bad).validate()

    def test_overlap_dropout_is_whole_hand_and_preserves_targets(self):
        p,t=demo_motion(180);p[:,1]=p[:,0]
        cfg=Config(fit_extent=.25,randomize_errors=False,randomize_hand_shape=False,
                   false_positive_prob=0,hand_dropout_prob=0,burst_dropout_prob=0,
                   overlap_dropout_max=1)
        d=simulate(p,t,cfg)
        baseline=simulate(p,t,replace(cfg,overlap_dropout_max=0))
        self.assertTrue(d['overlap_dropout_mask'].any())
        self.assertFalse(d['input_mask'][d['overlap_dropout_mask']].any())
        np.testing.assert_array_equal(d['target_xyz'],baseline['target_xyz'])
        np.testing.assert_array_equal(d['query_time'],baseline['query_time'])
        np.testing.assert_array_equal(d['overlap_dropout_mask'],simulate(p,t,cfg)['overlap_dropout_mask'])
        for camera in range(3):
            for frame in range(1,len(d['query_time'])):
                if d['selected_frame'][frame,camera]==d['selected_frame'][frame-1,camera]:
                    np.testing.assert_array_equal(d['overlap_dropout_mask'][frame,camera],d['overlap_dropout_mask'][frame-1,camera])

    def test_background_false_positives_are_input_only_and_held(self):
        p,t=demo_motion(120)
        cfg=Config(false_positive_prob=1,false_positive_frames_min=3,false_positive_frames_max=3,
                   randomize_errors=False,fit_extent=.25)
        d=simulate(p,t,cfg)
        clean=simulate(p,t,replace(cfg,false_positive_prob=0))
        np.testing.assert_array_equal(d['target_xyz'],clean['target_xyz'])
        np.testing.assert_array_equal(d['target_mask'],clean['target_mask'])
        np.testing.assert_array_equal(d['query_time'],clean['query_time'])
        self.assertFalse(clean['false_positive_mask'].any())
        self.assertTrue(d['false_positive_mask'].any())
        self.assertTrue(d['input_mask'][d['false_positive_mask']].all())
        fake_pixels=d['pixels'][d['false_positive_mask']]
        self.assertTrue(np.isfinite(fake_pixels).all())
        self.assertTrue(((fake_pixels>=0)&(fake_pixels<[cfg.width,cfg.height])).all())
        np.testing.assert_allclose(np.linalg.norm(d['ray_directions'][d['false_positive_mask']],axis=-1),1,atol=1e-6)
        for camera in range(3):
            for frame in range(1,len(d['query_time'])):
                if d['selected_frame'][frame,camera]==d['selected_frame'][frame-1,camera]:
                    np.testing.assert_array_equal(d['pixels'][frame,camera],d['pixels'][frame-1,camera])
        np.testing.assert_array_equal(d['pixels'],simulate(p,t,cfg)['pixels'])
        for bad in (-.1,1.1,float('nan')):
            with self.assertRaises(ValueError): replace(cfg,false_positive_prob=bad).validate()

    def test_false_positive_prefers_empty_slot_and_background(self):
        from gigahands_sim import background_false_positives
        cfg=Config(false_positive_prob=1,false_positive_frames_min=3,false_positive_frames_max=3)
        uv=np.full((9,2,21,2),[160.,120.])
        detected=np.tile([True,False],(9,1))
        result,mask=background_false_positives(uv,detected,cfg,np.random.default_rng(9))
        self.assertFalse(mask[:,0].any())
        self.assertTrue(mask[:,1].all())
        np.testing.assert_array_equal(result[:,0],uv[:,0])
        self.assertGreater(np.linalg.norm(result[0,1].mean(0)-[160,120]),50)
        self.assertLess(np.max(np.abs(result[1,1]-result[0,1])),3)

    def test_distance_dropout_curve_and_camera_specificity(self):
        from gigahands_sim import hand_dropout_probabilities
        cfg=Config()
        distances=np.array([0,25,50,75,100,150,200])
        world=np.zeros((len(distances),2,21,3))
        world[:,0,0,0]=distances/40
        world[:,1,0,0]=.25
        p,b,d=hand_dropout_probabilities(world,np.zeros(3),cfg)
        np.testing.assert_allclose(p[:,0],[.01,.015,.02,.05,.10,.20,.20])
        np.testing.assert_allclose(b,p/5)
        np.testing.assert_allclose(d[:,0],distances)
        other=hand_dropout_probabilities(world,np.array([1,0,0]),cfg)[0]
        self.assertFalse(np.array_equal(p,other))
        off=hand_dropout_probabilities(world,np.zeros(3),replace(cfg,distance_dropout_enabled=False))[0]
        np.testing.assert_allclose(off,.05)
        rng=np.random.default_rng(123)
        draws=rng.random((100000,len(distances)))<p[:,0]
        np.testing.assert_allclose(draws.mean(0),p[:,0],atol=.004)
        with self.assertRaises(ValueError):replace(cfg,dropout_distance_cm=[50,0]).validate()

    def test_no_individual_dropout_and_subpixel_jitter(self):
        p,t=demo_motion(90);p[:]=p[0]
        cfg=Config(fit_extent=.25,randomize_errors=False,randomize_hand_shape=False,
                   position_std=0,angle_std_deg=0,hand_dropout_prob=0,burst_dropout_prob=0,
                   missing_prob=1,pixel_std=.1,correlated_pixel_std=.1,outlier_prob=0)
        noisy=simulate(p,t,cfg)
        clean=simulate(p,t,replace(cfg,missing_prob=0,pixel_std=0,correlated_pixel_std=0))
        np.testing.assert_array_equal(noisy['input_mask'],clean['input_mask'])
        selected=noisy['selected_frame']>=0
        self.assertTrue(noisy['input_mask'][selected].all())
        delta=(noisy['pixels']-clean['pixels'])[noisy['input_mask']]
        self.assertGreater(delta.std(),.05)
        self.assertLess(delta.std(),.2)
        np.testing.assert_array_equal(noisy['target_xyz'],clean['target_xyz'])
        self.assertEqual(Config().error_ranges['missing_prob'],[0,0])
        self.assertEqual(Config().error_ranges['outlier_prob'],[0,0])

    def test_own_query_times_and_workspace_windows(self):
        p,t=demo_motion(90)
        p[40:45,0,1,0]=3
        cfg=Config(query_sync_camera3=False, center=False, randomize_hand_shape=False)
        d=simulate(p,t,cfg)
        self.assertGreater(len(d['window_starts']),0)
        for start in d['window_starts']:
            self.assertFalse(start<=44 and start+cfg.window>40)
        i=1;start=d['window_starts'][i]
        x,mask,*_=training_window(d,i)
        np.testing.assert_array_equal(x,d['features'][start:start+cfg.window].reshape(-1,14))
        self.assertEqual(window_frame_offsets(d,i)[-1],0)
        self.assertGreater(json.loads(d['metadata'])['workspace']['excluded_windows'],0)
        with self.assertRaises(ValueError): simulate(p,t,replace(cfg,workspace_policy='strict'))

    def test_bursts_and_contact_protection(self):
        cfg=Config(burst_dropout_prob=1,burst_frames_min=3,burst_frames_max=3)
        self.assertTrue(hand_burst_mask(12,cfg,np.random.default_rng(1)).all())
        self.assertFalse(hand_burst_mask(12,replace(cfg,burst_dropout_prob=0),np.random.default_rng(1)).any())
        p,_=demo_motion(90);p,_=normalize(p,cfg)
        changed,meta=augment_hand_shape(p,replace(cfg,contact_sensitive=True,contact_augmentation_factor=0))
        np.testing.assert_allclose(changed,p,atol=1e-14)
        self.assertEqual(meta['augmentation_strength'],0)

    def test_jitter_correlation_quality(self):
        p,t=demo_motion(300);p[:]=p[0]
        cfg=Config(fit_extent=.25,randomize_errors=False,randomize_hand_shape=False,
                   hand_dropout_prob=0,burst_dropout_prob=0,pixel_std=0,outlier_prob=0,missing_prob=0,
                   position_std=0,angle_std_deg=0,correlated_pixel_std=1,correlated_pixel_rho=.9,
                   inference_stall_prob=.2,inference_stall_ms=[20,20])
        d=simulate(p,t,cfg);e=d['hand_timing_events']
        residual=e[:,4]-np.array(cfg.hand_inference_ms)[e[:,3].astype(int)]
        self.assertGreater(residual.std(),2)
        camera=0;sel=d['selected_frame'][:,camera];indices=np.flatnonzero((sel>=0)&np.r_[True,sel[1:]!=sel[:-1]])
        good=indices[d['input_mask'][indices,camera,0,0]]
        uv=d['pixels'][good,camera,0,0,0]
        self.assertGreater(np.corrcoef(uv[:-1],uv[1:])[0,1],.3)
        q=clip_quality(d)
        self.assertEqual(sum(q['hand_camera_count']),len(d['query_time'])*2)
        self.assertGreater(sum(q['inference_time_hist']),0)

    def test_resume_errors_provenance_and_disjoint_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);archive=root/'source.tar.gz'
            raw=json.dumps(demo_motion(90)[0].tolist()).encode()
            with tarfile.open(archive,'w:gz') as tar:
                for i in range(6):
                    content=raw if i<5 else b'bad json'
                    info=tarfile.TarInfo(f'p{i:03d}-task/keypoints_3d/000.json');info.size=len(content)
                    tar.addfile(info,io.BytesIO(content))
            cfg=Config()
            out=root/'out';run(archive,out,cfg,max_bytes=1)
            state=run(archive,out,cfg,max_bytes=None,resume=True)
            self.assertEqual((state['clips_done'],state['clips_failed']),(5,1))
            rows=[json.loads(x) for x in (out/'manifest.jsonl').read_text().splitlines()]
            self.assertTrue(all('file' in r and 'quality' in r for r in rows))
            self.assertEqual(len((out/'failures.jsonl').read_text().splitlines()),1)
            splits=json.loads((out/'split.json').read_text())['participants']
            self.assertEqual(set(splits.values()),{'train','val','test'})
            previous=(out/'manifest.jsonl').read_bytes()
            # Simulate an interrupted append; atomic receipts recover the index.
            with (out/'manifest.jsonl').open('ab') as stream:stream.write(b'{unfinished')
            run(archive,out,cfg,max_bytes=None,resume=True)
            self.assertEqual((out/'manifest.jsonl').read_bytes(),previous)
            with self.assertRaises(ValueError):run(archive,out,replace(cfg,seed=99),resume=True)
            with (out/rows[0]['file']).open('r+b') as stream:stream.write(b'bad!')
            with self.assertRaises(ValueError):run(archive,out,cfg,resume=True)
