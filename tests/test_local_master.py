import threading
from types import SimpleNamespace
import unittest

from master import local_inference


class LocalMasterTests(unittest.TestCase):
    def test_local_frame_goes_to_slot_one_without_touching_remote_frames(self):
        stop = threading.Event()
        remote2, remote3, frame = object(), object(), object()
        slots = [None, remote2, remote3]
        closed = []
        meta = {'id': 1, 'landmarks': [], 'inference_ms': 1}

        class Tracker:
            def __init__(self, args):
                self.args = args

            def __enter__(self):
                return self

            def read(self):
                stop.set()
                return frame, meta

            def __exit__(self, *exc):
                closed.append(True)

        local_inference(SimpleNamespace(id=1, fps=15), slots, threading.Lock(), stop, Tracker)
        self.assertIs(slots[0][0], frame)
        self.assertIs(slots[0][1], meta)
        self.assertIs(slots[1], remote2)
        self.assertIs(slots[2], remote3)
        self.assertEqual(closed, [True])

    def test_local_failure_does_not_stop_remote_receivers(self):
        stop = threading.Event()
        slots = [None, object(), object()]
        before = list(slots)

        def broken_tracker(args):
            raise RuntimeError('Camera unavailable')

        with self.assertLogs(level='ERROR'):
            local_inference(SimpleNamespace(id=1, fps=15), slots, threading.Lock(), stop, broken_tracker)
        self.assertEqual(slots, before)
        self.assertFalse(stop.is_set())


if __name__ == '__main__':
    unittest.main()
