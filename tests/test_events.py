"""hand_tracking.events: event acceptance, latest rays and the query-time anchor."""
import unittest
from model_helpers import asynchronous, synthetic_events
import torch
from hand_tracking.events import accepted_captures, event_anchor, latest_rays, make_events, query_anchor


def accepted_by_running_max(camera, capture, present):
    """Reference: per-camera running maximum (the original cummax formulation)."""
    accepted = torch.zeros_like(present)
    for camera_id in range(3):
        belongs = present & (camera == camera_id)
        maximum = torch.where(belongs, capture, float('-inf')).cummax(dim=1).values
        previous = torch.cat((torch.full_like(maximum[:,:1], float('-inf')), maximum[:,:-1]), dim=1)
        accepted |= belongs & (capture > previous)
    return accepted


class AcceptanceTests(unittest.TestCase):
    def test_ignores_padding_and_tracks_each_camera(self):
        camera = torch.tensor([[0,1,0,0,1,0,1]])
        capture = torch.tensor([[.1,.1,99.,.1,.2,.09,.15]], dtype=torch.float64)
        present = torch.tensor([[True,True,False,True,True,True,True]])
        expected = torch.tensor([[True,True,False,False,True,False,False]])
        self.assertTrue(torch.equal(accepted_captures(camera, capture, present), expected))

    def test_pairwise_rule_matches_running_maximum(self):
        generator = torch.Generator().manual_seed(4)
        camera = torch.randint(0, 3, (64,40), generator=generator)
        capture = torch.randint(0, 12, (64,40), generator=generator).double()*.01    # many ties
        present = torch.rand(64,40, generator=generator) > .2
        torch.testing.assert_close(accepted_captures(camera, capture, present),
                                   accepted_by_running_max(camera, capture, present))


class LatestRayTests(unittest.TestCase):
    def test_frame_without_detections_replaces_older_rays(self):
        base = torch.randn(2,21,3)*.1
        inputs = list(synthetic_events(lambda t: base, [-.3,-.3,-.3,-.1], [0,1,2,2]))
        inputs[1] = inputs[1].clone()
        inputs[1][0,-1] = False                                                  # camera 2's newest frame: nothing seen
        _, _, mask, capture = latest_rays(make_events(*inputs), torch.tensor([3]), .3)
        self.assertFalse(mask[0,0,2].any())                                       # camera 2's older ray is replaced
        self.assertTrue(mask[0,0,:2].all())
        self.assertAlmostEqual(capture[0,0,2].item(), -.1, places=5)
        # The sample of that event still triangulates from cameras 0 and 1.
        anchor = event_anchor(*inputs, inputs[4][:,-1:], lookback_s=0.)[0,0]
        torch.testing.assert_close(anchor, base, atol=1e-5, rtol=0)

    def test_stale_capture_arriving_late_does_not_replace_newer(self):
        new = torch.randn(2,21,3)*.1
        old = new+.2
        # Sorted by capture: c0@-.3, c0@-.25 (stale), c0@-.1, c1@-.1; the -.25 capture arrives last.
        inputs = list(synthetic_events(lambda t: new if t > -.2 else old, [-.3,-.25,-.1,-.1], [0,0,0,1], delay=0.))
        inputs[4] = torch.tensor([[-.3,-.05,-.1,-.1]])
        order = torch.argsort(inputs[4][0])
        inputs = [value[:,order] for value in inputs]
        anchor = event_anchor(*inputs, inputs[4][:,-1:], lookback_s=0.)[0,0]
        torch.testing.assert_close(anchor, new, atol=1e-5, rtol=0)


class AnchorTests(unittest.TestCase):
    def test_static_hold_and_hand_fallback(self):
        base = torch.randn(2,21,3)*.1
        inputs = list(synthetic_events(lambda t: base, *asynchronous()))
        inputs[1] = inputs[1].clone()
        inputs[1][:,:,1] = False
        inputs[1][:,:,1,3] = True                                                 # hand 1: only joint 3
        anchor = event_anchor(*inputs, torch.tensor([[0.]]))[0,0]
        torch.testing.assert_close(anchor[0], base[0], atol=1e-5, rtol=0)
        torch.testing.assert_close(anchor[1], base[1,3].expand(21,3), atol=1e-5, rtol=0)
        # Nothing arrived yet: origin; samples older than the hold span are not used.
        first = float(inputs[4][0,0])
        self.assertTrue((event_anchor(*inputs, torch.tensor([[first-.01]])) == 0).all())
        self.assertTrue((event_anchor(*inputs, torch.tensor([[1.]]), lookback_s=0.) == 0).all())

    def test_samples_on_a_line_are_extrapolated_to_the_query(self):
        base = torch.randn(2,21,3)*.1
        velocity = torch.randn(2,21,3)*.3
        stamps = torch.linspace(-.3, -.05, 6)                                     # sample capture times
        point = (base+velocity*stamps[:,None,None,None])[None]                    # [1,E,2,21,3]
        ok = torch.ones(1,6,2,21, dtype=torch.bool)
        stamp = stamps[None,:,None,None].expand(1,6,2,21)
        arrival = stamps[None]+.06
        present = torch.ones(1,6, dtype=torch.bool)
        query = torch.tensor([[.02]])
        moved, flags = query_anchor(point, ok, stamp, arrival, present, query, .2, .3)
        held, _ = query_anchor(point, ok, stamp, arrival, present, query, 0., .3)
        torch.testing.assert_close(moved[0,0], base+velocity*.02, atol=1e-5, rtol=0)
        torch.testing.assert_close(held[0,0], point[0,-1], atol=1e-6, rtol=0)
        self.assertTrue(flags[0,0,...,4].bool().all())
        # Samples of events not yet arrived at the query are not used.
        early, _ = query_anchor(point, ok, stamp, arrival, present, torch.tensor([[float(arrival[0,2])]]), 0., .3)
        torch.testing.assert_close(early[0,0], point[0,2], atol=1e-6, rtol=0)


if __name__ == '__main__':
    unittest.main()
