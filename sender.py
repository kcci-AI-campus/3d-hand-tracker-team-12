"""Run on each camera Pi. USB camera by default; --backend picamera2 for CSI."""
import argparse
import logging
from pathlib import Path
import socket
import threading
import time

from protocol import pack_frame, read_exact


class LatestCamera:
    def __init__(self, args, cv2):
        self.cv2 = cv2
        self.backend = args.backend
        self.condition = threading.Condition()
        self.frame = None
        self.sequence = 0
        self.error = None
        self.stop = threading.Event()
        if args.backend == 'picamera2':
            from picamera2 import Picamera2
            self.camera = Picamera2()
            # Picamera2 RGB888 arrays use B,G,R byte order, compatible with OpenCV.
            self.camera.configure(self.camera.create_video_configuration(
                main={'size': (320, 240), 'format': 'RGB888'},
                controls={'FrameRate': args.fps}, buffer_count=4))
            self.camera.start()
        else:
            self.camera = cv2.VideoCapture(args.camera)
            if not self.camera.isOpened():
                self.camera.release()
                raise RuntimeError('Cannot open USB camera')
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            self.camera.set(cv2.CAP_PROP_FPS, args.fps)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()

    def _capture(self):
        try:
            while not self.stop.is_set():
                if self.backend == 'picamera2':
                    frame = self.camera.capture_array('main')
                else:
                    ok, frame = self.camera.read()
                    if not ok:
                        raise RuntimeError('Camera frame capture failed')
                captured_ns = time.time_ns()
                frame = self.cv2.resize(frame, (320, 240))
                with self.condition:
                    self.frame = (frame, captured_ns)
                    self.sequence += 1
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()

    def get(self, previous):
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.error or self.sequence != previous, timeout=5)
            if self.error:
                raise RuntimeError('Camera stopped') from self.error
            if not ready:
                raise RuntimeError('Camera timed out')
            return self.sequence, self.frame

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            logging.warning('Camera read is blocked; process exit will release device')
            return
        if self.backend == 'picamera2':
            self.camera.stop()
            self.camera.close()
        else:
            self.camera.release()


def add_camera_arguments(parser):
    parser.add_argument('--backend', choices=('usb', 'picamera2'), default='usb')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--model', default='models/hand_landmarker.task')
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--hands', type=int, choices=(1, 2), default=2)


class HandTracker:
    """Shared camera/inference pipeline for the local master and remote senders."""
    def __init__(self, args):
        self.args = args
        self.camera = None
        self.detector = None
        self.sequence, self.timestamp = 0, -1

    def __enter__(self):
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        self.cv2, self.mp = cv2, mp
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.args.model),
            running_mode=vision.RunningMode.VIDEO, num_hands=self.args.hands)
        self.detector = vision.HandLandmarker.create_from_options(options)
        try:
            self.camera = LatestCamera(self.args, cv2)
        except BaseException:
            self.detector.close()
            raise
        return self

    def read(self):
        self.sequence, (frame, captured_ns) = self.camera.get(self.sequence)
        self.timestamp = max(self.timestamp + 1, time.monotonic_ns() // 1_000_000)
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        start = time.monotonic()
        result = self.detector.detect_for_video(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), self.timestamp)
        inference_ms = (time.monotonic() - start) * 1000
        meta = {'id': self.args.id, 'seq': self.sequence, 'capture_unix_ns': captured_ns,
                'inference_ms': inference_ms,
                'landmarks': [[[p.x, p.y, p.z] for p in hand] for hand in result.hand_landmarks],
                'world_landmarks': [[[p.x, p.y, p.z] for p in hand] for hand in result.hand_world_landmarks],
                'handedness': [hand[0].category_name for hand in result.handedness]}
        return frame, meta

    def __exit__(self, *exc):
        try:
            self.camera.close()
        finally:
            self.detector.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--master', required=True)
    parser.add_argument('--id', type=int, choices=(2, 3), required=True)
    parser.add_argument('--base-port', type=int, default=5000)
    add_camera_arguments(parser)
    parser.add_argument('--quality', type=int, default=75)
    args = parser.parse_args()
    if not 1 <= args.fps <= 60 or not 1 <= args.quality <= 100 or not 1 <= args.base_port <= 65533:
        parser.error('fps: 1..60, quality: 1..100, base-port: 1..65533')
    if not Path(args.model).is_file():
        parser.error('Model missing: run python download_model.py')
    import cv2
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    try:
        with HandTracker(args) as tracker:
            while True:
                try:
                    with socket.create_connection((args.master, args.base_port + args.id - 1), timeout=3) as sock:
                        sock.settimeout(5)
                        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        logging.info('Connected as Pi %d', args.id)
                        while True:
                            start = time.monotonic()
                            frame, meta = tracker.read()
                            ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                            if not ok:
                                raise RuntimeError('JPEG encoding failed')
                            sock.sendall(pack_frame(meta, encoded.tobytes()))
                            if read_exact(sock, 1) != b'K':
                                raise ConnectionError('Bad ACK')
                            time.sleep(max(0, 1 / args.fps - (time.monotonic() - start)))
                except (OSError, ConnectionError) as exc:
                    logging.warning('Network disconnected: %s; retry in 1s', exc)
                    time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
