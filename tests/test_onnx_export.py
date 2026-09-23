"""hand_tracking.export and export_onnx.py: the ONNX graph matches PyTorch at other sizes.
Skipped unless requirements-export.txt is installed; one export takes ~20 s on CPU."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from model_helpers import randomize_heads, sim_clip, small

MISSING = [name for name in ('onnx', 'onnxscript', 'onnxruntime') if importlib.util.find_spec(name) is None]
if MISSING:
    raise unittest.SkipTest(f'Install requirements-export.txt for ONNX tests (missing {", ".join(MISSING)})')

import torch
import export_onnx
from hand_tracking.checkpoints import save_checkpoint
from hand_tracking.config import SamplingConfig
from hand_tracking.export import compare_onnx, example_inputs, export_onnx as export_model
from hand_tracking.model import HandTransformer


class OnnxExportTests(unittest.TestCase):
    def test_checkpoint_export_matches_pytorch(self):
        torch.manual_seed(0)
        model = randomize_heads(HandTransformer(small()).eval(), std=.05)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            clip_folder = folder/'clip'
            clip_folder.mkdir()
            sim_clip(clip_folder)
            checkpoint = folder/'model.pt'
            save_checkpoint(checkpoint, dict(model=model.state_dict(), model_config=vars(model.config).copy(),
                                             sampling_config=vars(SamplingConfig(64)).copy()))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                export_onnx.main(['--checkpoint', str(checkpoint), '--output', str(folder/'model.onnx'),
                                  '--check-input', str(clip_folder/'a.npz')])
            report = json.loads(output.getvalue())
            self.assertEqual(set(report['max_abs_difference']), {'synthetic', 'npz'})
            self.assertLess(max(report['max_abs_difference'].values()), 1e-4)
            # Other batch, event and query counts than the traced example and the check above.
            self.assertLess(compare_onnx(folder/'model.onnx', model, example_inputs(1, 23, 7)), 1e-4)

    def test_motion_fit_is_not_exportable(self):
        with self.assertRaisesRegex(ValueError, 'anchor_motion_fit'):
            export_model(HandTransformer(small(anchor_motion_fit=True)), 'unused.onnx')


if __name__ == '__main__':
    unittest.main()
