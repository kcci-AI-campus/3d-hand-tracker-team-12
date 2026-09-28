"""Test the master without slave Pis: replay a dataset clip's camera 0 and 1 frames as UV2 packets
in real time (the u,v the slaves would have sent, the capture -> send delay they would have
measured). Run on the development PC from the repository root, pointing at the master:

    python hand_tracker_master/tools/fake_slaves.py --master 192.168.0.10
    python hand_tracker_master/tools/fake_slaves.py --master 127.0.0.1 --loops 3

Needs numpy and the repository (hand_tracking.data reads the clip)."""
import argparse
from pathlib import Path
import socket
import struct
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hand_tracking.data import read_clip  # noqa: E402

HEADER = struct.Struct('!4sBBHIII')     # "HUV2", version, camera, flags, sequence, capture->send us, infer us


def packet(camera, sequence, delay_s, uv):
    return HEADER.pack(b'HUV2', 2, camera, 0, sequence & 0xffffffff, int(delay_s*1e6), 0)+np.asarray(uv, '>f4').tobytes()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--master', required=True, help='Master Pi IPv4')
    p.add_argument('--ports', default='5001,5002', help='Master UDP ports of camera 0 and 1')
    p.add_argument('--clip', default=str(ROOT/'exports'/'gigahands_pi3_overlap'/'val'/'clip_00001.npz'))
    p.add_argument('--loops', type=int, default=1)
    args = p.parse_args()
    ports = [int(port) for port in args.ports.split(',')]
    clip = read_clip(args.clip)
    order = [i for i in np.argsort(clip['event_arrival'], kind='stable') if clip['event_camera'][i] < 2]
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sequence, sent = [0, 0], 0
    duration = float(clip['event_arrival'][order[-1]]-clip['event_arrival'][order[0]])
    print(f'Replaying {len(order)} frames of cameras 0/1 ({duration:.1f} s) x{args.loops} -> {args.master}:{ports}', flush=True)
    for _ in range(args.loops):
        start, first = time.monotonic(), float(clip['event_arrival'][order[0]])
        for i in order:
            camera = int(clip['event_camera'][i])
            # The dataset's arrival is when the master got the frame: send then, carrying its age.
            time.sleep(max(0., start+float(clip['event_arrival'][i])-first-time.monotonic()))
            valid = clip['event_valid'][i].astype(bool)
            uv = np.where(valid[..., None], clip['event_features'][i][..., :2], -1.).reshape(-1)
            delay = float(clip['event_arrival'][i]-clip['event_capture'][i])
            sock.sendto(packet(camera, sequence[camera], delay, uv), (args.master, ports[camera]))
            sequence[camera] += 1
            sent += 1
    print('Sent', sent, 'packets')


if __name__ == '__main__':
    main()
