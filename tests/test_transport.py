import socket
import threading
import unittest

from master import validate
from protocol import HEADER, MAGIC, MAX_JPEG, pack_frame, read_exact, receive_frame


def metadata(node=1):
    return {'id': node, 'inference_ms': 12.5, 'landmarks': [[[0.1, 0.2, -0.01]] * 21]}


class TransportTests(unittest.TestCase):
    def test_fragmented_frames_and_ack(self):
        sender, receiver = socket.socketpair()
        errors = []
        def transmit():
            try:
                with sender:
                    sender.settimeout(2)
                    for seq in range(2):
                        packet = pack_frame(dict(metadata(), seq=seq), b'fake-jpeg')
                        for offset in range(0, len(packet), 7):
                            sender.sendall(packet[offset:offset + 7])
                        self.assertEqual(read_exact(sender, 1), b'K')
            except Exception as exc:
                errors.append(exc)
        worker = threading.Thread(target=transmit)
        worker.start()
        with receiver:
            receiver.settimeout(2)
            for seq in range(2):
                meta, jpeg = receive_frame(receiver)
                validate(meta, 1)
                self.assertEqual(meta['seq'], seq)
                self.assertEqual(jpeg, b'fake-jpeg')
                receiver.sendall(b'K')
        worker.join(timeout=3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])

    def test_rejects_oversize_before_reading_body(self):
        a, b = socket.socketpair()
        with a, b:
            b.settimeout(1)
            a.sendall(HEADER.pack(MAGIC, 10, MAX_JPEG + 1))
            with self.assertRaises(ValueError):
                receive_frame(b)

    def test_truncated_connection(self):
        a, b = socket.socketpair()
        a.sendall(b'HN')
        a.close()
        with b:
            with self.assertRaises(ConnectionError):
                receive_frame(b)

    def test_invalid_landmarks_and_id(self):
        for data in [dict(metadata(), id=2), dict(metadata(), landmarks=[[]]),
                     dict(metadata(), landmarks=[[[float('nan'), 0, 0]] * 21]),
                     dict(metadata(), inference_ms=float('inf'))]:
            with self.assertRaises(ValueError):
                validate(data, 1)

    def test_empty_and_nonfinite_payload_rejected(self):
        for meta, jpeg in [(metadata(), b''), ({'value': float('nan')}, b'jpeg')]:
            with self.assertRaises(ValueError):
                pack_frame(meta, jpeg)


if __name__ == '__main__':
    unittest.main()
