"""Compare every original TFLite output with deployment-version ncnn C++ output."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from convert_models import OUTPUTS

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', type=Path, required=True)
    args = parser.parse_args()
    directory = ROOT / 'work' / 'parity'
    directory.mkdir(exist_ok=True)
    photo = cv2.cvtColor(cv2.imread(str(ROOT.parent / 'testdata/hand.jpg')), cv2.COLOR_BGR2RGB)
    report = {}
    for model, labels in OUTPUTS.items():
        interpreter = Interpreter(model_path=str(ROOT / 'source' / (model+'.tflite')), num_threads=2)
        interpreter.allocate_tensors()
        input_info = interpreter.get_input_details()[0]
        size = int(input_info['shape'][1])
        real = cv2.resize(photo, (size,size)).astype(np.float32) / 255
        samples = {'black':np.zeros_like(real), 'white':np.ones_like(real),
                   'random':np.random.default_rng(42).random(real.shape,dtype=np.float32),
                   'hand':real, 'hand_rotated':np.rot90(real).copy(), 'hand_mirrored':real[:,::-1].copy()}
        report[model] = {}
        for name, nhwc in samples.items():
            interpreter.set_tensor(input_info['index'], nhwc[None].copy())
            interpreter.invoke()
            references = [interpreter.get_tensor(d['index']) for d in interpreter.get_output_details()]
            input_path = directory / f'{model}_{name}_input.f32'
            nhwc.transpose(2,0,1).copy().tofile(input_path)
            output_prefix = directory / f'{model}_{name}'
            subprocess.run([str(args.probe.resolve()),str(ROOT/'converted'/model),str(input_path),str(size),
                            str(output_prefix),'palm' if model=='hand_detector' else 'hand'],check=True,capture_output=True)
            errors = {}
            for label, expected in zip(labels, references):
                actual = np.fromfile(str(output_prefix)+'_'+label+'.f32',np.float32).reshape(expected.shape)
                np.testing.assert_allclose(actual,expected,atol=2e-3,rtol=2e-3,err_msg=f'{model}/{name}/{label}')
                diff = np.abs(actual-expected)
                errors[label] = {'max_abs':float(diff.max()),'mean_abs':float(diff.mean()),'shape':list(actual.shape)}
            report[model][name] = errors
            print(model,name,{k:round(v['max_abs'],8) for k,v in errors.items()},flush=True)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report['_validation'] = {
        'atol':2e-3, 'rtol':2e-3,
        'runtime':subprocess.check_output([str(args.probe.resolve()), '--version'],text=True).strip(),
        'conversion_toolchain':json.loads((ROOT/'conversion_report.json').read_text())['_toolchain'],
        'probe_sha256':sha(args.probe),
        'converted_sha256':{p.name:sha(p) for p in sorted((ROOT/'converted').glob('*')) if p.suffix in ('.param','.bin')},
        'source_sha256':{p.name:sha(p) for p in sorted((ROOT/'source').glob('*.tflite'))},
    }
    (ROOT / 'parity_report.json').write_text(json.dumps(report,indent=2))

if __name__ == '__main__': main()
