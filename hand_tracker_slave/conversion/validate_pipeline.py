"""Integration checks against real C++ inference, without opening a camera."""
from pathlib import Path
import json
import re
import subprocess
import cv2
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--app', type=Path, required=True)
args = parser.parse_args()
project = Path(__file__).resolve().parent.parent
exe = args.app.resolve()
out = project / 'conversion/work/image_checks'
out.parent.mkdir(exist_ok=True)
out.mkdir(exist_ok=True)
sample = cv2.imread(str(project / 'testdata/hand.jpg'))
square = cv2.resize(sample, (480, 480))
images = {
    'normal': (square, 1),
    'clockwise': (cv2.rotate(square, cv2.ROTATE_90_CLOCKWISE), 1),
    'upside_down': (cv2.rotate(square, cv2.ROTATE_180), 1),
    'mirror': (cv2.flip(square, 1), 1),
    'two_hands': (np.concatenate([square, cv2.flip(square, 1)], axis=1), 2),
    'portrait': (cv2.copyMakeBorder(square, 180, 180, 30, 30, cv2.BORDER_CONSTANT, value=(255,)*3), 1),
    'landscape': (cv2.copyMakeBorder(square, 30, 30, 180, 180, cv2.BORDER_CONSTANT, value=(255,)*3), 1),
    'black': (np.zeros((240, 320, 3), np.uint8), 0),
    'white': (np.full((240, 320, 3), 255, np.uint8), 0),
}
report, coordinates = {}, {}
for name, (image, count) in images.items():
    path = out / f'{name}.png'
    cv2.imwrite(str(path), image)
    result = subprocess.run([str(exe), '--models', str(project / 'models'), '--image', str(path),
                             '--expect-hands', str(count)], capture_output=True, text=True, timeout=60)
    (out / f'{name}.txt').write_text(result.stdout + result.stderr)
    report[name] = {'exit_code': result.returncode, 'summary': result.stdout.splitlines()[:2]}
    if result.returncode:
        raise RuntimeError(f'{name}: {result.stdout}\n{result.stderr}')
    pts = re.findall(r'^\d+: \[([^\]]+)\]', result.stdout, re.M)
    coordinates[name] = np.array([[float(x) for x in p.split(',')] for p in pts])
normal = coordinates['normal'][:, :2]
for name in ['clockwise', 'upside_down', 'mirror']:
    expected = normal.copy()
    if name == 'clockwise':
        expected[:, 0], expected[:, 1] = 479 - normal[:, 1], normal[:, 0]
    elif name == 'upside_down':
        expected = 479 - normal
    else:
        expected[:, 0] = 479 - normal[:, 0]
    error = np.linalg.norm(coordinates[name][:, :2] - expected, axis=1)
    report[name]['mean_pixel_error_vs_transformed_normal'] = float(error.mean())
    if error.mean() > 25:
        raise RuntimeError(f'Coordinate mapping regression: {name}, mean error {error.mean()}')
(project / 'conversion/image_checks.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
