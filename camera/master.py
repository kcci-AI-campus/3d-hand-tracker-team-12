"""Pi 1: local inference plus two TCP receivers; three views on the main thread."""
import argparse
import logging
import math
from pathlib import Path
import socket
import threading
import time

from protocol import receive_frame
from sender import HandTracker, add_camera_arguments

EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7),
         (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13),
         (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
         (17, 18), (18, 19), (19, 20))


def validate(meta, node):
    if meta.get('id') != node:
        raise ValueError('Sender ID does not match port')
    hands = meta.get('landmarks')
    if not isinstance(hands, list) or len(hands) > 2:
        raise ValueError('Invalid hands')
    for hand in hands:
        if not isinstance(hand, list) or len(hand) != 21:
            raise ValueError('Invalid landmark count')
        for p in hand:
            if not isinstance(p, list) or len(p) != 3 or any(
                type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 100 for v in p):
                raise ValueError('Invalid coordinate')
    ms = meta.get('inference_ms')
    if type(ms) not in (int, float) or not math.isfinite(ms) or ms < 0:
        raise ValueError('Invalid inference time')


def receiver(server, node, slots, lock, stop, cv2, np):
    while not stop.is_set():
        try:
            conn, address = server.accept()
        except socket.timeout:
            continue
        with conn:
            conn.settimeout(5)
            logging.info('Pi %d connected: %s', node, address)
            try:
                while not stop.is_set():
                    meta, jpeg = receive_frame(conn)
                    validate(meta, node)
                    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None or frame.shape != (240, 320, 3):
                        raise ValueError('Expected 320x240 JPEG')
                    with lock:
                        slots[node - 1] = (frame, meta, time.monotonic())
                    conn.sendall(b'K')
            except (OSError, ValueError, ConnectionError, cv2.error) as exc:
                logging.warning('Pi %d disconnected: %s', node, exc)


def local_inference(args, slots, lock, stop, tracker_factory=HandTracker):
    """Publish raw local frames directly, without JPEG or loopback TCP overhead."""
    try:
        with tracker_factory(args) as tracker:
            while not stop.is_set():
                start = time.monotonic()
                frame, meta = tracker.read()
                with lock:
                    slots[0] = (frame, meta, time.monotonic())
                stop.wait(max(0, 1 / args.fps - (time.monotonic() - start)))
    except Exception:
        # Keep the two remote views alive even if the local camera fails.
        logging.exception('Pi 1 local inference stopped; restart master to retry')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bind', default='0.0.0.0')
    parser.add_argument('--base-port', type=int, default=5000)
    parser.add_argument('--stale-seconds', type=float, default=2)
    parser.add_argument('--separate-windows', action='store_true')
    add_camera_arguments(parser)
    parser.set_defaults(id=1)
    args = parser.parse_args()
    if not 1 <= args.base_port <= 65533 or not math.isfinite(args.stale_seconds) or args.stale_seconds <= 0:
        parser.error('Invalid port or stale threshold')
    if not 1 <= args.fps <= 60:
        parser.error('fps: 1..60')
    if not Path(args.model).is_file():
        parser.error('Model missing: run python camera/download_model.py')
    import cv2
    import numpy as np
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    slots, lock, stop = [None] * 3, threading.Lock(), threading.Event()
    servers, threads = [], []
    names = [f'Pi {i}' for i in (1, 2, 3)] if args.separate_windows else ['Three Pi Hands']
    try:
        for node in (2, 3):
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            servers.append(server)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((args.bind, args.base_port + node - 1))
            server.listen(1)
            server.settimeout(0.5)
        for node, server in zip((2, 3), servers):
            thread = threading.Thread(target=receiver, args=(server, node, slots, lock, stop, cv2, np), daemon=True)
            thread.start()
            threads.append(thread)
        local_thread = threading.Thread(target=local_inference, args=(args, slots, lock, stop), daemon=True)
        local_thread.start()
        threads.append(local_thread)
        for name in names:
            cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        logging.info('Pi 1 local inference; TCP %d..%d for Pi 2/3; Q/ESC exits', args.base_port + 1, args.base_port + 2)
        while True:
            with lock:
                current = list(slots)
            panels = []
            for index, item in enumerate(current):
                panel = np.zeros((240, 320, 3), dtype=np.uint8)
                label = f'Pi {index + 1}: waiting'
                if item:
                    frame, meta, received = item
                    age = time.monotonic() - received
                    if age <= args.stale_seconds:
                        panel = frame.copy()
                        for hand in meta['landmarks']:
                            points = [(max(0, min(319, round(p[0] * 320))), max(0, min(239, round(p[1] * 240)))) for p in hand]
                            for a, b in EDGES:
                                cv2.line(panel, points[a], points[b], (0, 255, 0), 1)
                            for point in points:
                                cv2.circle(panel, point, 2, (0, 0, 255), -1)
                        label = f'Pi {index + 1}: infer {meta["inference_ms"]:.0f}ms'
                    else:
                        label = f'Pi {index + 1}: STALE {age:.1f}s'
                cv2.rectangle(panel, (0, 0), (319, 24), (0, 0, 0), -1)
                cv2.putText(panel, label, (7, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                panels.append(panel)
            if args.separate_windows:
                for name, panel in zip(names, panels):
                    cv2.imshow(name, panel)
            else:
                cv2.imshow(names[0], np.hstack(panels))
            if cv2.waitKey(15) & 0xFF in (27, ord('q')):
                break
            if any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1 for name in names):
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=6)
        for server in servers:
            server.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
