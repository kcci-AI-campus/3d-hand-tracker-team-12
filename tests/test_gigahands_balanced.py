import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from gigahands_balanced import balanced_plan, run
from gigahands_sim import Config, demo_motion


class BalancedTests(unittest.TestCase):
    def test_round_robin_and_participant_split(self):
        records = [dict(source=f'p{p:03d}-{a}/{n}.json', participant=f'p{p:03d}', activity=a)
                   for p in range(1,11) for a in ('a','b') for n in range(3)]
        plan, split = balanced_plan(records)
        self.assertEqual((plan,split), balanced_plan(records))
        self.assertEqual(sum(v=='val' for v in split.values()),2)
        for start in range(0,len(plan),10):
            self.assertEqual(len({r['participant'] for r in plan[start:start+10]}),10)
        self.assertEqual(len({r['source'] for r in plan}),len(records))
        for p in split:
            personal = [r for r in plan if r['participant']==p]
            self.assertNotEqual(personal[0]['activity'],personal[1]['activity'])

    def test_stream_generation_cleanup_and_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); archive=root/'motions.tar.gz'
            raw=json.dumps(demo_motion(20)[0].tolist()).encode()
            with tarfile.open(archive,'w:gz') as tar:
                for p in range(1,6):
                    info=tarfile.TarInfo(f'p{p:03d}-task/keypoints_3d/000.json'); info.size=len(raw)
                    tar.addfile(info,io.BytesIO(raw))
            state=run(archive,root/'out',Config(),max_bytes=None)
            self.assertEqual(state['status'],'completed')
            self.assertEqual((state['train'],state['val']),(4,1))
            rows=[json.loads(line) for line in (root/'out/manifest.jsonl').read_text().splitlines()]
            self.assertEqual(len({r['source'] for r in rows}),5)
            self.assertEqual(len(list((root/'out').rglob('*.npz'))),5)
            self.assertFalse(list((root/'out').glob('tmp*')))
            limited=run(archive,root/'limited',Config(),max_bytes=1)
            self.assertEqual(limited['status'],'size_limit')
            self.assertEqual(limited['clips_done'],1)
