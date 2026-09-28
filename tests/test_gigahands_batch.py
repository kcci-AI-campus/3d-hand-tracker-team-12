import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import numpy as np
from gigahands_batch import run
from gigahands_sim import Config, demo_motion


class BatchTests(unittest.TestCase):
    def test_limit_finishes_three_variants_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root/'motion.tar.gz'
            payload = json.dumps(demo_motion(90)[0].tolist()).encode()
            with tarfile.open(archive, 'w:gz') as tar:
                for name in ('keypoints_3d/a.json', 'keypoints_3d/b.json'):
                    info = tarfile.TarInfo(name); info.size = len(payload)
                    tar.addfile(info, io.BytesIO(payload))
            output = root/'output'
            state = run(archive, output, Config(), max_bytes=1)
            self.assertEqual(state['status'], 'size_limit')
            self.assertEqual(state['clips_done'], 1)
            self.assertEqual(state['files'], 3)
            rows = [json.loads(line) for line in (output/'manifest.jsonl').read_text().splitlines()]
            self.assertEqual(len({row['pose_seed'] for row in rows}), 3)
            self.assertEqual(len({row['seed'] for row in rows}), 1)
            self.assertEqual(sum((output/row['file']).stat().st_size for row in rows), state['bytes'])
            poses = []
            for row in rows:
                with np.load(output/row['file']) as data:
                    poses.append(data['actual_origins'].copy())
                    self.assertTrue(json.loads(str(data['metadata']))['source'].endswith('::keypoints_3d/a.json'))
            self.assertFalse(np.array_equal(poses[0], poses[1]))
            with self.assertRaises(FileExistsError):
                run(archive, output, Config())
            state = run(archive, output, Config(), max_bytes=10_000_000, resume=True)
            self.assertEqual(state['status'], 'completed')
            self.assertEqual(state['files'], 6)
            self.assertEqual(state['clips_done'], 2)
            rows = [json.loads(line) for line in (output/'manifest.jsonl').read_text().splitlines()]
            self.assertEqual(len({row['file'] for row in rows}), 6)
