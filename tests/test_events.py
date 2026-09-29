"""hand_tracking.events: event acceptance and slot selection."""
import unittest
from model_helpers import ORIGINS, asynchronous, synthetic_events
import torch
from hand_tracking.events import accepted_captures, make_events, select_slots


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


class SlotTests(unittest.TestCase):
    def test_each_cameras_latest_events_within_the_span_in_arrival_order(self):
        captures, cameras = asynchronous(duration=.6)
        events = make_events(*synthetic_events(lambda t: ORIGINS.mean(0).expand(2, 21, 3), captures, cameras))
        query = torch.tensor([[-.2, 0.]])
        per, span = 3, .25
        slot, valid = select_slots(events, query, per, span)
        self.assertEqual(slot.shape, (1, 2, 3*per))
        for q in range(2):
            usable = (events['arrival'][0] <= query[0, q]+1e-6) & (events['capture'][0] >= query[0, q]-span)
            expected = []
            for camera in range(3):
                mine = torch.nonzero(usable & (events['camera'][0] == camera))[:, 0]
                expected += mine[-per:].tolist()
            chosen = slot[0, q][valid[0, q]].tolist()
            self.assertEqual(chosen, sorted(expected))                               # arrival order
            self.assertFalse(valid[0, q, :3*per-len(chosen)].any())                  # padding first


if __name__ == '__main__':
    unittest.main()
