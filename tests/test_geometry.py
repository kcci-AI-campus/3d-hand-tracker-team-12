"""hand_tracking.geometry: triangulation, ray residuals and the closed-form 3x3 helpers."""
import unittest
from model_helpers import ORIGINS
import torch
from hand_tracking.geometry import cross3, ray_residuals, solve3, triangulate


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

    def test_ray_residuals_are_angular_misses(self):
        point = torch.tensor([.1,-.2,.05])
        d = torch.nn.functional.normalize(point-ORIGINS, dim=-1)
        torch.testing.assert_close(ray_residuals(point, ORIGINS, d), torch.zeros(3), atol=1e-6, rtol=0)
        shifted = point+torch.tensor([0.,0.,.01])
        depth = ((shifted-ORIGINS)*d).sum(-1)
        miss = ((shifted-ORIGINS)-depth[:, None]*d).norm(dim=-1)
        torch.testing.assert_close(ray_residuals(shifted, ORIGINS, d), miss/depth)

    def test_closed_form_solve_and_cross_match_torch(self):
        torch.manual_seed(0)
        a = torch.randn(64,3,3, dtype=torch.float64)
        a = a@a.transpose(-1,-2)+torch.eye(3, dtype=torch.float64)*.1
        b = torch.randn(64,3, dtype=torch.float64)
        torch.testing.assert_close(solve3(a, b), torch.linalg.solve(a, b))
        torch.testing.assert_close(cross3(a[:,0], a[:,1]), torch.linalg.cross(a[:,0], a[:,1]))


if __name__ == '__main__':
    unittest.main()
