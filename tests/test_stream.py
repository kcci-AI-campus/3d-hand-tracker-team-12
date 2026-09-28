"""hand_tracking.stream with HandLite: a stream equals forward() event by event."""
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from model_helpers import linear_events, push_all, randomize_heads, sim_clip, small, synthetic_events
import torch
from hand_tracking.data import make_sample
from hand_tracking.lite import HandLite, LiteStream


class StreamEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_matches_forward_at_any_query_time(self):
        torch.manual_seed(10)
        with tempfile.TemporaryDirectory() as folder:
            clip = sim_clip(folder)
        model = randomize_heads(HandLite(small(blocks=2)).eval(), std=.05)
        stream, pushed, checked = LiteStream(model), 0, 0
        order = np.argsort(clip['event_arrival'], kind='stable')
        for window in range(0, len(clip['window_starts']), 4):
            sample = make_sample(clip, window, context_s=model.config.context_s)
            start = int(clip['window_starts'][window])
            last = float(clip['query_time'][start+clip['window']-1])
            for query in (last-.021, last):                       # between query frames and at one
                while pushed < len(order) and clip['event_arrival'][order[pushed]] <= query+1e-9:
                    i = order[pushed]
                    pushed += 1
                    stream.push(clip['event_camera'][i], clip['event_features'][i], clip['event_valid'][i],
                                clip['event_capture'][i], clip['event_arrival'][i])
                if not pushed:
                    continue
                inputs = [sample[key][None] for key in ('event_features', 'event_valid', 'event_camera',
                                                         'event_capture', 'event_arrival', 'event_present')]
                with torch.no_grad():
                    expected = model(*inputs, torch.tensor([[query-last]]))[0,0]
                torch.testing.assert_close(stream.query(query), expected, atol=1e-4, rtol=1e-4)
                checked += 1
        self.assertGreaterEqual(checked, 5)
        with self.assertRaises(ValueError):
            stream.query(float(clip['event_arrival'][order[pushed-1]])-1)

    def test_equal_arrival_times_match_batch(self):
        torch.manual_seed(12)
        model = randomize_heads(HandLite(small()).eval())
        base, velocity = torch.randn(2,21,3)*.1, torch.randn(2,21,3)*.3
        times = np.arange(-.5, 0, 1/17.1)
        inputs = synthetic_events(lambda t: base+velocity*t, [t for t in times for _ in range(3)],
                                  [c for _ in times for c in range(3)])
        stream, last = LiteStream(model), None
        for i in range(inputs[0].shape[1]):
            arrival = float(inputs[4][0,i])
            if last is not None and arrival == last:
                stream.query(arrival)                                  # query between equal-time pushes
            push_all(stream, inputs, [i])
            last = arrival
        with torch.no_grad():
            expected = model(*inputs, torch.tensor([[last]]))[0,0]
        torch.testing.assert_close(stream.query(last), expected, atol=1e-4, rtol=1e-4)

    def test_stale_and_duplicate_captures_match_padded_batch(self):
        torch.manual_seed(21)
        model = randomize_heads(HandLite(small()).eval())
        base = torch.randn(2,21,3)*.1
        inputs = list(synthetic_events(lambda t: base+t*.8, [0,0,0,.05,.1,.1,.1,.1], [0,1,2,0,0,1,2,0], delay=0.))
        # Last two camera-0 packets duplicate/reverse its accepted .1 capture.
        inputs[4] = torch.tensor([[.01,.01,.01,.15,.11,.11,.11,.16]])
        order = torch.argsort(inputs[4][0], stable=True)
        inputs = [value[:,order] for value in inputs]
        stream = LiteStream(model)
        for i in range(inputs[0].shape[1]):
            push_all(stream, inputs, [i])
            stream.flush()
        expected = stream.query(.17)
        # Arbitrary padding values cannot change event acceptance or attention.
        padded = [torch.cat((value, value[:,:2]), dim=1) for value in inputs]
        padded[5][:,-2:] = False
        batched = [value.expand(2, *value.shape[1:]).clone() for value in padded]
        with torch.inference_mode():
            actual = model(*batched, torch.full((2,1), .17))
        torch.testing.assert_close(actual[:,0], expected[None].expand(2,-1,-1,-1), atol=1e-5, rtol=1e-5)

    def test_pruned_history_and_large_absolute_clock_without_calibration(self):
        model = HandLite(small(calibration_head=False)).eval()
        randomize_heads(model, std=.05)
        base = torch.randn(2,21,3)*.1
        times = [step*.06 for step in range(50) for _ in range(3)]
        inputs = synthetic_events(lambda t: base+t*.01, times, [0,1,2]*50)
        stream = LiteStream(model)
        epoch = 1_700_000_000.
        for i in range(0, len(times), 3):
            for j in range(i, i+3):
                stream.push(int(inputs[2][0,j]), inputs[0][0,j], inputs[1][0,j],
                            epoch+float(inputs[3][0,j]), epoch+float(inputs[4][0,j]))
            actual = stream.query(epoch+float(inputs[4][0,i+2]))
        with torch.inference_mode():
            expected = model(*inputs, inputs[4][:,-1:])[0,0]
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        self.assertLess(len(stream), len(times))


class StreamBufferTests(unittest.TestCase):
    def test_stale_packet_is_dropped(self):
        new = torch.randn(2,21,3)*.1
        inputs = list(synthetic_events(lambda t: new, [-.3,-.25,-.1,-.1], [0,0,0,1], delay=0.))
        inputs[4] = torch.tensor([[-.3,-.05,-.1,-.1]])
        order = torch.argsort(inputs[4][0])
        inputs = [value[:,order] for value in inputs]
        self.assertEqual(push_all(LiteStream(HandLite(small())), inputs), [True,True,True,False])

    def test_failed_flush_can_be_retried_without_losing_events(self):
        model = HandLite(small()).eval()
        stream = LiteStream(model)
        inputs = synthetic_events(lambda t: torch.zeros(2,21,3), [0,0,0], [0,1,2])
        for i in range(3):
            stream.push(i, inputs[0][0,i], inputs[1][0,i], 0., .06)
        with patch.object(model, 'encode_events', side_effect=RuntimeError('probe')):
            with self.assertRaisesRegex(RuntimeError, 'probe'):
                stream.flush()
        self.assertEqual(len(stream), 3)
        stream.flush()
        self.assertEqual(len(stream), 3)
        self.assertTrue(torch.isfinite(stream.query(.06)).all())

    def test_owns_reused_tensor_and_numpy_buffers(self):
        torch.manual_seed(31)
        inputs, _, _ = linear_events(frames=12)
        model = randomize_heads(HandLite(small()).eval(), std=.01)
        features, valid, camera, capture, arrival, _ = (value[0] for value in inputs)
        for numpy_buffer in (False, True):
            reference, stream = LiteStream(model), LiteStream(model)
            buffer = np.empty(features.shape[1:], np.float32) if numpy_buffer else torch.empty_like(features[0])
            mask_buffer = np.empty(valid.shape[1:], bool) if numpy_buffer else torch.empty_like(valid[0])
            for i in range(len(camera)):
                args = (camera[i].item(), capture[i].item(), arrival[i].item())
                reference.push(args[0], features[i].clone(), valid[i].clone(), *args[1:])
                if numpy_buffer:
                    np.copyto(buffer, features[i].numpy())
                    np.copyto(mask_buffer, valid[i].numpy())
                else:
                    buffer.copy_(features[i])
                    mask_buffer.copy_(valid[i])
                stream.push(args[0], buffer, mask_buffer, *args[1:])
                # External mutation after push must not affect any stored event.
                if numpy_buffer:
                    buffer.fill(float('nan'))
                    mask_buffer.fill(False)
                else:
                    buffer.fill_(float('nan'))
                    mask_buffer.fill_(False)
                torch.testing.assert_close(stream.query(args[2]), reference.query(args[2]))


if __name__ == '__main__':
    unittest.main()
