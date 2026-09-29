"""Publish only the ncnn bytes that passed the current parity report."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent

def main():
    report = json.loads((ROOT/'parity_report.json').read_text())
    validated = report['_validation']
    if len(validated['converted_sha256']) != 4: raise RuntimeError('Expected two model pairs')
    for folder,key in [('converted','converted_sha256'),('source','source_sha256')]:
        for name,expected in validated[key].items():
            if hashlib.sha256((ROOT/folder/name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'{name} changed since numerical validation; rerun the probe tests')
    models = ROOT.parent/'models'
    models.mkdir(exist_ok=True)
    for name in validated['converted_sha256']: shutil.copyfile(ROOT/'converted'/name,models/name)
    (models/'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name,digest in validated['converted_sha256'].items()), encoding='utf-8', newline='\n')
    for name in ('palm-lite-op.param','palm-lite-op.bin','hand_lite-op.param','hand_lite-op.bin'):
        (models/name).unlink(missing_ok=True)
    manifest = {'official_source':json.loads((ROOT/'official_source.json').read_text()),
                'ncnn_sha256':validated['converted_sha256'], 'precision':'FP32 (source FP16 constants expanded exactly)',
                'tools':{name:importlib.metadata.version(name) for name in ('torch','tflite','ai-edge-litert','numpy')},
                'ncnn_runtime':validated['runtime'],
                'converter':validated['conversion_toolchain'],
                'output_blobs':{'hand_detector':['regressors','scores'],
                                'hand_landmarks_detector':['landmarks','presence','handedness','world_landmarks']}}
    (models/'model_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Published four validated ncnn model files and model_manifest.json')

if __name__ == '__main__': main()
