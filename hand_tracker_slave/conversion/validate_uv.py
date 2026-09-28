"""Receive real image inference from the C++ sender; check all 84 wire values."""
import argparse
import json
from pathlib import Path
import re
import socket
import struct
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--app', type=Path, required=True)
args = parser.parse_args()
project = Path(__file__).resolve().parent.parent
report = {}
for name, mirrored in [('normal', False), ('normal', True), ('two_hands', False), ('black', False)]:
    image = project / 'conversion/work/image_checks' / f'{name}.png'
    width, height = (960, 480) if name == 'two_hands' else (320, 240) if name == 'black' else (480, 480)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(('127.0.0.1', 0))
        receiver.settimeout(5)
        command = [str(args.app.resolve()), '--models', str(project/'models'), '--image', str(image),
                   '--send-uv']
        if mirrored:
            command.append('--mirrored')
        result = subprocess.run(command, input=f'127.0.0.1\n{receiver.getsockname()[1]}\n', capture_output=True, text=True, check=True, timeout=30)
        assert 'Master IPv4:' in result.stdout and 'Master UDP port [5001]:' in result.stdout
        assert 'TX: 1  Failed: 0  Bytes: 336  Last socket error: 0' in result.stdout
        packet, _ = receiver.recvfrom(65535)
    assert len(packet) == 336
    actual = struct.unpack('!84f', packet)
    expected, best = [-1.0]*84, [-1.0]*2
    for section in result.stdout.split('Palm confidence: ')[1:]:
        presence, right = re.search(r'Presence: ([\d.e+-]+).*probability: ([\d.e+-]+)', section).groups()
        right = float(right) if mirrored else 1-float(right)
        slot = int(right >= .5)
        confidence = float(presence)*(right if slot else 1-right)
        if confidence <= best[slot]:
            continue
        best[slot] = confidence
        points = re.findall(r'^\d+: \[([^\]]+)\]', section, re.M)
        assert len(points) == 21
        for j, point in enumerate(points):
            x, y, _ = map(float, point.split(','))
            uv = (x/width, y/height)
            expected[slot*42+j*2:slot*42+j*2+2] = uv if all(0 <= v <= 1 for v in uv) else (-1, -1)
    assert all(abs(a-b) < 1e-5 for a, b in zip(actual, expected))
    report[f'{name}_mirrored_{mirrored}'] = {'bytes': len(packet), 'valid_joints': sum(actual[i] >= 0 for i in range(0, 84, 2))}
(project/'conversion/uv_checks.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
