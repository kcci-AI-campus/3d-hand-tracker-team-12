"""hand_tracking.geometry: triangulation, rotations, Gauss-Newton calibration, motion fit."""
import math
import unittest
from model_helpers import ORIGINS, asynchronous, linear_events, synthetic_events
import torch
from hand_tracking.events import event_anchor
from hand_tracking.geometry import cross3, rodrigues, solve3, triangulate


def rodrigues_reference(w):
    """Float64 textbook formula with the unit axis (undefined at zero)."""
    theta = w.norm(dim=-1)[...,None,None]
    k = w/w.norm(dim=-1, keepdim=True)
    skew = torch.zeros(*w.shape[:-1], 3, 3, dtype=w.dtype)
    skew[...,0,1], skew[...,0,2], skew[...,1,2] = -k[...,2], k[...,1], -k[...,0]
    skew = skew-skew.transpose(-1,-2)
    return torch.eye(3, dtype=w.dtype)+torch.sin(theta)*skew+(1-torch.cos(theta))*skew@skew


class TriangulationTests(unittest.TestCase):
    def test_exact_degenerate_and_optional_gate(self):
        point = torch.tensor([.1,-.2,.05])
        d = torch.nn.functional.normalize(point-ORIGINS, dim=-1)
        all_rays = torch.ones(3, dtype=torch.bool)
        result, valid = triangulate(ORIGINS, d, all_rays)
        self.assertTrue(valid)
        torch.testing.assert_close(result, point, atol=1e-5, rtol=0)
        self.assertFalse(triangulate(ORIGINS, -d, all_rays)[1])                   # behind the cameras
        result, valid = triangulate(ORIGINS, d, torch.tensor([True,False,False]))
        self.assertFalse(valid)
        self.assertTrue((result == 0).all())
        bent = d.clone()
        bent[2] = torch.nn.functional.normalize(bent[2]+torch.tensor([.5,0.,0.]), dim=-1)
        self.assertTrue(triangulate(ORIGINS, bent, all_rays)[1])                   # no gate by default
        self.assertFalse(triangulate(ORIGINS, bent, all_rays, .18)[1])
        parallel = triangulate(torch.tensor([[0.,0,0],[0,.1,0]]), torch.tensor([[1.,0,0],[1,0,0]]),
                               torch.tensor([True,True]))
        self.assertFalse(parallel[1])

    def test_closed_form_solve_and_cross_match_torch(self):
        torch.manual_seed(0)
        a = torch.randn(64,3,3, dtype=torch.float64)
        a = a@a.transpose(-1,-2)+torch.eye(3, dtype=torch.float64)*.1
        b = torch.randn(64,3, dtype=torch.float64)
        torch.testing.assert_close(solve3(a, b), torch.linalg.solve(a, b))
        torch.testing.assert_close(cross3(a[:,0], a[:,1]), torch.linalg.cross(a[:,0], a[:,1]))


class RotationTests(unittest.TestCase):
    def test_rodrigues_accuracy_and_gradient_at_zero(self):
        w = torch.randn(500,3, dtype=torch.float64)*torch.logspace(-7, 0, 500, dtype=torch.float64)[:,None]
        self.assertLess((rodrigues(w.float()).double()-rodrigues_reference(w)).abs().max().item(), 1e-6)
        zero = torch.zeros(3, requires_grad=True)
        vector = torch.tensor([.3,-.2,.5])
        (rodrigues(zero)@vector).sum().backward()
        # d(R v)/dw at w=0 is -[v]x, so the summed gradient is v x (1,1,1).
        torch.testing.assert_close(zero.grad, torch.linalg.cross(vector, torch.ones(3)))
        self.assertAlmostEqual(torch.det(rodrigues(torch.tensor([0.,0.,math.pi/2]))).item(), 1., places=5)


class GaussNewtonTests(unittest.TestCase):
    def test_baseline_recovers_relative_camera_error(self):
        torch.manual_seed(5)
        # A static, widely spread point set: rich multi-view geometry whatever rays are combined.
        base = torch.randn(2,21,3)*.3
        inputs = list(synthetic_events(lambda t: base, *asynchronous()))
        params = torch.zeros(3,6)
        params[1:,:3] = torch.randn(2,3)*.05
        params[1:,3:] = torch.randn(2,3)*.05
        features = inputs[0].clone()
        for i, camera in enumerate(inputs[2][0]):
            features[0,i,...,5:8] = features[0,i,...,5:8]@rodrigues(params[camera,:3])  # nominal = R^T true
            features[0,i,...,2:5] = features[0,i,...,2:5]-params[camera,3:]
        inputs[0] = features
        query = inputs[4][:,-1:]

        def shape_error(steps):
            """Error after the best similarity transform: the common rig motion is unobservable."""
            a = event_anchor(*inputs, query, lookback_s=0., calibration_steps=steps)[0,0].reshape(-1,3)
            b = base.reshape(-1,3)
            a, b = a-a.mean(0), b-b.mean(0)
            u, sv, vt = torch.linalg.svd(a.T@b)
            d = torch.diag(torch.tensor([1., 1., torch.sign(torch.det(u@vt))]))
            return ((sv*torch.diag(d)).sum()/a.square().sum()*a@(u@d@vt)-b).norm(dim=-1).mean().item()

        self.assertLess(shape_error(5), shape_error(0)*.01)


class MotionFitTests(unittest.TestCase):
    def test_handles_asynchronous_cameras(self):
        base = torch.randn(2,21,3)*.1
        velocity = torch.tensor([1.,.3,0.])
        at = lambda t: base+velocity*t
        inputs = synthetic_events(at, *asynchronous())
        query = float(inputs[4][0,-1])
        truth = at(torch.tensor(query))
        static = event_anchor(*inputs, torch.tensor([[query]]))[0,0]
        fitted = event_anchor(*inputs, torch.tensor([[query]]), motion_fit=True)[0,0]
        fitted_error = (fitted-truth).norm(dim=-1).max().item()
        static_error = (static-truth).norm(dim=-1).max().item()
        self.assertLess(fitted_error, 1e-3)
        self.assertLess(fitted_error, static_error*.2)                         # async samples are biased

    def test_rejects_one_outlier_per_camera(self):
        inputs, query, target = linear_events()
        for camera in (None, 0, 1, 2):
            with self.subTest(camera=camera):
                features = inputs[0].clone()
                if camera is not None:
                    last = int((inputs[2][0] == camera).nonzero().max())
                    features[0,last,...,5:8] = torch.nn.functional.normalize(
                        features[0,last,...,5:8]+torch.tensor([.5,0.,0.]), dim=-1)
                anchor = event_anchor(features, *inputs[1:], query[None,-1:], motion_fit=True)
                self.assertLess((anchor[0,0]-target[-1]).norm(dim=-1).max().item()*300, .05)   # mm at 30 cm/unit

    def test_single_camera_gives_no_anchor(self):
        inputs, query, _ = linear_events()
        present = inputs[2] == 0
        anchor = event_anchor(*inputs[:5], present, query[None,-1:], motion_fit=True)
        self.assertTrue((anchor == 0).all())


if __name__ == '__main__':
    unittest.main()
