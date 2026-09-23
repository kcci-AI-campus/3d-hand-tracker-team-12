import unittest
import numpy as np
from gigahands_gui import camera_frame
from gigahands_sim import look_at, rotation


class CameraRenderingTests(unittest.TestCase):
    def test_frustum_matches_projection_after_pose_error(self):
        origin = np.array([1.02, -1.01, .53])
        r = rotation(np.deg2rad([4., -6., 3.])) @ look_at(origin, np.zeros(3))
        k = np.array([300., 310., 157., 118.])
        corners = camera_frame(origin, r, k, 320, 240)
        camera = (corners-origin) @ r.T
        uv = camera[:, :2]/camera[:, 2:] * k[:2] + k[2:]
        np.testing.assert_allclose(uv, [[0, 0], [319, 0], [319, 239], [0, 239]], atol=1e-10)
        np.testing.assert_allclose(camera[:, 2], .45)


if __name__ == '__main__':
    unittest.main()
