"""End-to-end CLI on a tiny simulated dataset: train (and resume), evaluate, predict and export
(when ncnn/pnnx are installed) for both architectures."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from model_helpers import small  # noqa: F401  (skips the module without torch)
from training import evaluate, export, predict, train
from gigahands_sim import Config, demo_motion, simulate, save_dataset

TINY_LITEV3 = ['--dim', '32', '--heads', '4', '--blocks', '1', '--slots-per-camera', '4', '--dropout', '0']
TINY_DIRECT = ['--arch', 'direct', '--dim', '32', '--heads', '4', '--blocks', '1', '--fusion-blocks', '1',
               '--slots-per-camera', '4', '--dropout', '0']


def tiny_dataset(root):
    """Two simulated clips (train/val) with a manifest, as the exporter writes them."""
    positions, times = demo_motion(30)
    data = simulate(positions, times, Config())
    root.mkdir(parents=True)
    rows = []
    for split, participant in (('train', 'p001'), ('val', 'p002')):
        save_dataset(root/f'{split}.npz', data)
        rows.append(dict(file=f'{split}.npz', split=split, participant=participant, source=split,
                         windows=len(data['window_starts'])))
    (root/'manifest.jsonl').write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')


def quietly(main, *argv):
    with contextlib.redirect_stdout(io.StringIO()) as output:
        main(list(map(str, argv)))
    return output.getvalue()


class ScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = Path(cls.folder.name)
        tiny_dataset(cls.root/'data')

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def train(self, run, *extra, model=TINY_LITEV3):
        return quietly(train.main, '--data', self.root/'data', '--output', run, '--epochs', 1,
                       '--batch-size', 2, '--windows-per-clip', 2, '--val-windows-per-clip', 2, '--threads', 2,
                       '--device', 'cpu', *model, *extra)

    def export_report(self, run):
        if any(importlib.util.find_spec(module) is None for module in ('ncnn', 'pnnx')):
            self.skipTest('Install requirements-export.txt to test training.export')
        # pnnx logs from native code; only the report goes through Python's stdout.
        return json.loads(quietly(export.main, '--checkpoint', run/'best.pt', '--output', run/'deploy',
                                  '--check-input', self.root/'data'/'val.npz', '--queries', 8))

    def evaluate_and_predict(self, run):
        quietly(evaluate.main, '--checkpoint', run/'best.pt', '--output', run/'model.json',
                '--data', self.root/'data', '--windows-per-clip', 2, '--batch-size', 2)
        evaluation = json.loads((run/'model.json').read_text())
        self.assertTrue(np.isfinite([evaluation['mpjpe_mm'], evaluation['mpjpe_world_mm']]).all())
        quietly(predict.main, '--checkpoint', run/'best.pt', '--input', self.root/'data'/'val.npz',
                '--output', run/'prediction.npz')
        with np.load(run/'prediction.npz') as prediction:
            self.assertEqual(prediction['predicted_xyz'].shape, (2,21,3))
            return evaluation, set(prediction.files)

    def test_litev3_train_resume_evaluate_predict_export(self):
        run = self.root/'litev3'
        self.assertIn('epoch=1/1', self.train(run))
        for name in ('run.json', 'metrics.jsonl', 'last.pt', 'best.pt'):
            self.assertTrue((run/name).is_file(), name)
        # Old run.json without a newer flag still resumes when that flag is at its default.
        options = json.loads((run/'run.json').read_text())
        del options['relative_weight']
        (run/'run.json').write_text(json.dumps(options))
        self.train(run, '--resume')
        self.train(run, '--resume', '--workers', 1, '--cache-dir', self.root/'cache')   # speed settings may change
        with self.assertRaisesRegex(ValueError, 'Resume settings'):
            self.train(run, '--resume', '--fit-span-s', '.3')
        evaluation, files = self.evaluate_and_predict(run)
        self.assertIsNotNone(evaluation['error_miss_mm'])
        self.assertIsNotNone(evaluation['kind_rate']['triangulated'])
        self.assertTrue({'anchor_xyz', 'anchor_kind', 'expected_error_mm', 'in_view_probability'} <= files)
        self.assertLess(self.export_report(run)['max_abs_difference'], 2e-3)

    def test_direct_train_evaluate_predict_export(self):
        run = self.root/'direct'
        self.assertIn('parameters=', self.train(run, model=TINY_DIRECT))
        self.train(run, '--resume', model=TINY_DIRECT)
        evaluation, files = self.evaluate_and_predict(run)
        self.assertIsNone(evaluation['error_miss_mm'])                         # no error, anchor or in-view head
        self.assertIsNone(evaluation['kind_rate']['triangulated'])
        self.assertNotIn('anchor_kind', files)
        self.assertLess(self.export_report(run)['max_abs_difference'], 2e-3)

    def test_invalid_settings_exit_through_the_parser(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.train(self.root/'bad', '--heads', '5')


if __name__ == '__main__':
    unittest.main()
