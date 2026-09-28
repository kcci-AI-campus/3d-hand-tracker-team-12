"""hand_tracking.geometry: triangulation and rotations."""
import math
import unittest
from model_helpers import ORIGINS
import torch
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
    def test_exact_and_degenerate(self):
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
        self.assertTrue(triangulate(ORIGINS, bent, all_rays)[1])                   # inconsistent, still a point
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


if __name__ == '__main__':
    unittest.main()
