import csv
from pathlib import Path
import socket
import tempfile
import threading
import sys
import unittest

# camera/ holds scripts that import each other by module name.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'camera'))
from latency_recorder import make_receiver_row, record_connection, summarize, estimate_clock
from protocol import pack_frame, read_exact


class LatencyTests(unittest.TestCase):
    def metadata(self):
        return dict(id=1, seq=1, capture_unix_ns=1_000_000_000,
                    tracker_done_unix_ns=1_010_000_000, send_prepare_unix_ns=1_020_000_000,
                    inference_ms=10., jpeg_encode_ms=2.)

    def test_clock_offset_is_not_assumed(self):
        meta = self.metadata()
        raw = make_receiver_row(meta, 4, 1_080_000_000, 500)
        self.assertEqual(raw['raw_capture_to_arrival_ms'], 80.)
        self.assertEqual(raw['capture_to_arrival_ms'], '')
        corrected = make_receiver_row(meta, 4, 1_080_000_000, 500, clock_offset_ms=10)
        self.assertEqual(corrected['capture_to_arrival_ms'], 90.)
        self.assertEqual(corrected['send_prepare_to_arrival_ms'], 70.)
        meta.update(clock_offset_ms=10.,clock_probe_rtt_ms=2.,clock_estimate_age_ms=100.)
        estimated = make_receiver_row(meta,4,1_080_000_000,500)
        self.assertEqual(estimated['clock_assumption'],'round_trip_estimate')
        self.assertEqual(estimated['capture_to_arrival_ms'],90.)

    def test_protocol_to_csv_and_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            listener = socket.socket(); listener.bind(('127.0.0.1', 0)); listener.listen(1)
            client = socket.create_connection(listener.getsockname())
            server, _ = listener.accept(); listener.close()
            worker = threading.Thread(target=record_connection, args=(server, folder, threading.Event()))
            worker.start()
            try:
                client.settimeout(3)
                offset,rtt,_ = estimate_clock(client)
                self.assertLess(abs(offset),20)
                self.assertGreaterEqual(rtt,0)
                client.sendall(pack_frame(self.metadata(), b'fake-jpeg'))
                self.assertEqual(read_exact(client,1), b'K')
            finally:
                client.close(); worker.join(6)
            self.assertFalse(worker.is_alive())
            paths = list(Path(folder).glob('*.csv'))
            self.assertEqual(len(paths),1)
            with paths[0].open(newline='',encoding='utf-8') as f: rows = list(csv.DictReader(f))
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['clock_assumption'],'unverified')
            self.assertEqual(rows[0]['capture_to_arrival_ms'],'')
            self.assertEqual(summarize(paths)['1']['inference_ms']['mean'],10.)


if __name__ == '__main__': unittest.main()
