"""Record real HandTracker/JPEG/TCP timings; see LATENCY.md for clock semantics."""
import argparse
import csv
import json
import logging
import math
from pathlib import Path
import socket
import statistics
import threading
import time
import uuid

from protocol import pack_frame, read_exact, receive_frame
from sender import HandTracker, add_camera_arguments


RECEIVER_FIELDS = [
    'session_id', 'camera_id', 'seq', 'capture_unix_ns', 'tracker_done_unix_ns',
    'send_prepare_unix_ns', 'arrival_unix_ns', 'arrival_monotonic_ns',
    'capture_interval_ms', 'arrival_interval_ms', 'skipped_capture_frames',
    'inference_ms', 'jpeg_encode_ms', 'jpeg_bytes', 'raw_capture_to_arrival_ms',
    'clock_assumption', 'sender_clock_offset_ms', 'capture_to_arrival_ms',
    'send_prepare_to_arrival_ms',
]
SENDER_FIELDS = [
    'session_id', 'camera_id', 'seq', 'capture_unix_ns', 'tracker_done_unix_ns',
    'send_prepare_unix_ns', 'ack_unix_ns', 'inference_ms', 'jpeg_encode_ms',
    'pack_ms', 'send_to_ack_ms', 'cycle_ms',
]


def open_csv(path, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open('x', newline='', encoding='utf-8')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    stream.flush()
    return stream, writer


def make_receiver_row(meta, jpeg_size, arrival_ns, arrival_mono_ns,
                      previous=None, clock_offset_ms=None):
    for key in ('id', 'seq', 'capture_unix_ns', 'tracker_done_unix_ns',
                'send_prepare_unix_ns'):
        if type(meta.get(key)) is not int or meta[key] < 0:
            raise ValueError('Invalid timing field: ' + key)
    for key in ('inference_ms', 'jpeg_encode_ms'):
        value = meta.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid timing field: ' + key)
    raw = (arrival_ns - meta['capture_unix_ns']) / 1e6
    row = dict(session_id='', camera_id=meta['id'], seq=meta['seq'],
               capture_unix_ns=meta['capture_unix_ns'],
               tracker_done_unix_ns=meta['tracker_done_unix_ns'],
               send_prepare_unix_ns=meta['send_prepare_unix_ns'],
               arrival_unix_ns=arrival_ns, arrival_monotonic_ns=arrival_mono_ns,
               capture_interval_ms='', arrival_interval_ms='', skipped_capture_frames='',
               inference_ms=meta['inference_ms'], jpeg_encode_ms=meta['jpeg_encode_ms'],
               jpeg_bytes=jpeg_size, raw_capture_to_arrival_ms=raw,
               clock_assumption='unverified', sender_clock_offset_ms='',
               capture_to_arrival_ms='', send_prepare_to_arrival_ms='')
    if previous is not None:
        if meta['id'] != previous['camera_id'] or meta['seq'] <= previous['seq']:
            raise ValueError('Camera ID changed or sequence did not increase')
        row['capture_interval_ms'] = (meta['capture_unix_ns'] - previous['capture_unix_ns']) / 1e6
        row['arrival_interval_ms'] = (arrival_mono_ns - previous['arrival_monotonic_ns']) / 1e6
        row['skipped_capture_frames'] = meta['seq'] - previous['seq'] - 1
    if clock_offset_ms is not None:
        # Offset = sender clock - receiver clock; subtract from sender timestamps.
        row['clock_assumption'] = 'user_supplied_offset'
        row['sender_clock_offset_ms'] = clock_offset_ms
        row['capture_to_arrival_ms'] = raw + clock_offset_ms
        row['send_prepare_to_arrival_ms'] = (
            arrival_ns - meta['send_prepare_unix_ns']) / 1e6 + clock_offset_ms
    return row


def record_connection(conn, output_dir, stop, clock_offset_ms=None):
    session = uuid.uuid4().hex
    previous = None
    stream = None
    try:
        conn.settimeout(5)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while not stop.is_set():
            meta, jpeg = receive_frame(conn)
            arrival_ns, arrival_mono_ns = time.time_ns(), time.monotonic_ns()
            row = make_receiver_row(meta, len(jpeg), arrival_ns, arrival_mono_ns,
                                    previous, clock_offset_ms)
            row['session_id'] = session
            # Match the existing ACK protocol; timestamp excludes disk writing.
            conn.sendall(b'K')
            if stream is None:
                path = Path(output_dir) / f"receiver_camera{meta['id']}_{session}.csv"
                stream, writer = open_csv(path, RECEIVER_FIELDS)
                logging.info('Recording %s', path)
            writer.writerow(row)
            stream.flush()
            previous = row
    except (OSError, ValueError, ConnectionError) as exc:
        logging.info('Receiver connection ended: %s', exc)
    finally:
        if stream is not None:
            stream.close()
        conn.close()


def run_receiver(args):
    stop = threading.Event()
    workers = []
    offset = 0.0 if args.clocks_synchronized else args.sender_clock_offset_ms
    if offset is None:
        logging.warning('Clocks unverified: corrected one-way latency columns remain empty')
    deadline = time.monotonic() + args.duration if args.duration else math.inf
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((args.bind, args.port))
            server.listen(8)
            server.settimeout(0.5)
            logging.info('Listening on %s:%d', args.bind, args.port)
            while time.monotonic() < deadline:
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    continue
                worker = threading.Thread(target=record_connection,
                                          args=(conn, args.output_dir, stop, offset))
                worker.start()
                workers = [item for item in workers if item.is_alive()]
                workers.append(worker)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for worker in workers:
            worker.join()


def run_sender(args):
    if not Path(args.model).is_file():
        raise SystemExit('Model missing: run python download_model.py')
    import cv2
    session = uuid.uuid4().hex
    path = Path(args.output_dir) / f'sender_camera{args.id}_{session}.csv'
    stream, writer = open_csv(path, SENDER_FIELDS)
    deadline = time.monotonic() + args.duration if args.duration else math.inf
    logging.info('Recording %s', path)
    try:
        with stream, HandTracker(args) as tracker:
            while time.monotonic() < deadline:
                try:
                    with socket.create_connection((args.host, args.port), timeout=3) as conn:
                        conn.settimeout(5)
                        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        while time.monotonic() < deadline:
                            start = time.monotonic_ns()
                            frame, meta = tracker.read()
                            tracker_done = time.time_ns()
                            encode_start = time.monotonic_ns()
                            ok, encoded = cv2.imencode('.jpg', frame,
                                                       [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                            if not ok:
                                raise RuntimeError('JPEG encoding failed')
                            jpeg = encoded.tobytes()
                            encode_ms = (time.monotonic_ns() - encode_start) / 1e6
                            meta.update(tracker_done_unix_ns=tracker_done,
                                        jpeg_encode_ms=encode_ms,
                                        send_prepare_unix_ns=time.time_ns())
                            pack_start = time.monotonic_ns()
                            packet = pack_frame(meta, jpeg)
                            send_start = time.monotonic_ns()
                            conn.sendall(packet)
                            if read_exact(conn, 1) != b'K':
                                raise ConnectionError('Bad ACK')
                            ack_mono, ack_unix = time.monotonic_ns(), time.time_ns()
                            writer.writerow(dict(
                                session_id=session, camera_id=args.id, seq=meta['seq'],
                                capture_unix_ns=meta['capture_unix_ns'],
                                tracker_done_unix_ns=tracker_done,
                                send_prepare_unix_ns=meta['send_prepare_unix_ns'],
                                ack_unix_ns=ack_unix, inference_ms=meta['inference_ms'],
                                jpeg_encode_ms=encode_ms, pack_ms=(send_start-pack_start)/1e6,
                                send_to_ack_ms=(ack_mono-send_start)/1e6,
                                cycle_ms=(ack_mono-start)/1e6))
                            stream.flush()
                            time.sleep(max(0, min(deadline - time.monotonic(),
                                1/args.fps - (time.monotonic_ns()-start)/1e9)))
                except (OSError, ConnectionError) as exc:
                    logging.warning('Disconnected: %s; reconnecting', exc)
                    time.sleep(max(0, min(1, deadline-time.monotonic())))
    except KeyboardInterrupt:
        pass


def percentile(values, fraction):
    position = (len(values)-1) * fraction
    lower = int(position)
    upper = min(lower+1, len(values)-1)
    return values[lower] + (values[upper]-values[lower]) * (position-lower)


def summarize(paths):
    groups = {}
    for path in paths:
        with Path(path).open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or 'camera_id' not in reader.fieldnames:
                raise ValueError(f'Not a timing CSV: {path}')
            for row in reader:
                for key, value in row.items():
                    if key.endswith('_ms') and value and key != 'sender_clock_offset_ms':
                        group = groups.setdefault(row['camera_id'], {})
                        group.setdefault(key, []).append(float(value))
    result = {}
    for camera, metrics in groups.items():
        result[camera] = {}
        for name, values in metrics.items():
            values.sort()
            result[camera][name] = dict(
                count=len(values), mean=statistics.mean(values),
                std_population=statistics.pstdev(values), min=values[0],
                p50=percentile(values, .5), p95=percentile(values, .95),
                p99=percentile(values, .99), max=values[-1])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='mode', required=True)
    receiver = subs.add_parser('receiver', help='Record received frames without GUI/decode')
    receiver.add_argument('--bind', default='0.0.0.0')
    receiver.add_argument('--port', type=int, default=5100)
    clock = receiver.add_mutually_exclusive_group()
    clock.add_argument('--clocks-synchronized', action='store_true')
    clock.add_argument('--sender-clock-offset-ms', type=float,
                       help='Sender clock minus receiver clock; applies to all clients')
    sender = subs.add_parser('sender', help='Run real camera/MediaPipe/JPEG pipeline')
    sender.add_argument('--host', required=True)
    sender.add_argument('--port', type=int, default=5100)
    sender.add_argument('--id', type=int, choices=(1, 2, 3), required=True)
    add_camera_arguments(sender)
    sender.set_defaults(fps=30)
    sender.add_argument('--quality', type=int, default=75)
    for command in (sender, receiver):
        command.add_argument('--output-dir', default='latency_logs')
        command.add_argument('--duration', type=float, default=0,
                             help='Seconds to run; 0 means until Ctrl+C')
    summary = subs.add_parser('summarize')
    summary.add_argument('paths', nargs='+', help='CSV files or directories')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if args.mode == 'summarize':
        paths = []
        for item in args.paths:
            path = Path(item)
            paths.extend(sorted(path.glob('*.csv')) if path.is_dir() else [path])
        print(json.dumps(summarize(paths), indent=2, allow_nan=False))
        return
    if not 1 <= args.port <= 65535 or not math.isfinite(args.duration) or args.duration < 0:
        parser.error('port: 1..65535; duration: finite and >= 0')
    if args.mode == 'sender':
        if not 1 <= args.fps <= 60 or not 1 <= args.quality <= 100:
            parser.error('fps: 1..60; quality: 1..100')
        run_sender(args)
    else:
        if args.sender_clock_offset_ms is not None and not math.isfinite(args.sender_clock_offset_ms):
            parser.error('Clock offset must be finite')
        run_receiver(args)


if __name__ == '__main__':
    main()
