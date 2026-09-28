"""PC side: receive the master Pi's H3D1 packets (3D joints of both hands) and print or plot them.

    python pc_receiver.py                 # UDP 6000, one status line per second
    python pc_receiver.py --plot          # live 3D view (needs matplotlib)
    python pc_receiver.py --save out.npz  # also record every packet

Packet (big-endian, 532 bytes): "H3D1", version 1, flags, reserved, sequence u32, query time f64 (s,
master clock), latency f32 (ms), cameras that saw the left/right hand (u8 each), reserved, then
126 float32: [left 21][right 21] joints x,y,z in cm in the rig frame (Y up). A hand seen by no
camera still has joints (the network's guess); treat it as absent.
Needs only numpy (and matplotlib for --plot)."""
import argparse
import socket
import struct
import time
import numpy as np

HEADER = struct.Struct('!4sBBHIdfBBH')      # 28 bytes
SIZE = HEADER.size+126*4
BONES = [(0, 1+4*f) for f in range(5)]+[(1+4*f+j, 2+4*f+j) for f in range(5) for j in range(3)]


def decode(data):
    """dict of one packet, or None for anything else."""
    if len(data) != SIZE:
        return None
    magic, version, _, _, sequence, query_time, latency, seen_left, seen_right, _ = HEADER.unpack_from(data)
    if magic != b'H3D1' or version != 1:
        return None
    joints = np.frombuffer(data, '>f4', 126, HEADER.size).astype(np.float32).reshape(2, 21, 3)
    return dict(sequence=sequence, time=query_time, latency_ms=latency, seen=(seen_left, seen_right), joints_cm=joints)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--port', type=int, default=6000)
    p.add_argument('--plot', action='store_true', help='Live 3D view of both hands')
    p.add_argument('--save', help='Record every packet to this .npz on exit (Ctrl+C)')
    p.add_argument('--count', type=int, default=0, help='Stop after this many packets (0: run until Ctrl+C)')
    args = p.parse_args()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', args.port))
    sock.settimeout(.2)
    print(f'H3D1 receiver on UDP {args.port}. Ctrl+C to exit.', flush=True)
    view = None
    if args.plot:
        import matplotlib.pyplot as plt
        plt.ion()
        figure = plt.figure(figsize=(7, 7))
        view = figure.add_subplot(111, projection='3d')
    records, received, bad, last_report, last = [], 0, 0, time.monotonic(), None
    lost = 0
    try:
        while not args.count or received < args.count:
            try:
                data, _ = sock.recvfrom(2048)
            except socket.timeout:
                if view is not None:
                    plt.pause(.001)
                continue
            packet = decode(data)
            if packet is None:
                bad += 1
                continue
            if last is not None:
                lost += max(0, (packet['sequence']-last['sequence']-1) & 0xffffffff if packet['sequence'] > last['sequence'] else 0)
            received += 1
            last = packet
            if args.save:
                records.append(packet)
            now = time.monotonic()
            if now-last_report >= 1:
                left, right = packet['joints_cm'][:, 0]
                print(f"#{packet['sequence']} latency {packet['latency_ms']:.0f}ms seen L/R {packet['seen'][0]}/{packet['seen'][1]} "
                      f"wrist L ({left[0]:.1f}, {left[1]:.1f}, {left[2]:.1f}) R ({right[0]:.1f}, {right[1]:.1f}, {right[2]:.1f}) cm "
                      f"| packets {received} lost {lost} bad {bad}", flush=True)
                last_report = now
            if view is not None:
                view.cla()
                for hand, color in enumerate(('tab:blue', 'tab:orange')):
                    style = '-' if packet['seen'][hand] else ':'
                    for a, b in BONES:
                        view.plot(*packet['joints_cm'][hand, [a, b]].T, color=color, linestyle=style)
                view.set(xlim=(-40, 40), ylim=(-40, 40), zlim=(-40, 40), xlabel='X (cm)', ylabel='Y (cm, up)', zlabel='Z (cm)',
                         title=f"latency {packet['latency_ms']:.0f} ms (dotted: unseen hand)")
                plt.pause(.001)
    except KeyboardInterrupt:
        pass
    if args.save and records:
        np.savez_compressed(args.save, sequence=np.array([r['sequence'] for r in records]),
                            time=np.array([r['time'] for r in records]), latency_ms=np.array([r['latency_ms'] for r in records]),
                            seen=np.array([r['seen'] for r in records]), joints_cm=np.stack([r['joints_cm'] for r in records]))
        print('Saved', len(records), 'packets to', args.save)
    print(f'Received {received}, lost {lost}, bad {bad}')


if __name__ == '__main__':
    main()
