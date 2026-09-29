// The nominal camera rig the model was trained with (models/rig.json, from the dataset NPZs) and the
// conversion of detected 2D joints into the model's event features.
#pragma once
#include <array>
#include <string>

#include "direct_runtime.hpp"

namespace hand_master {

struct Rig {
    int width = 320, height = 240;              // image size the u,v are normalised by
    double world_unit_cm = 40.;                  // 1 model world unit in cm
    std::array<std::array<double, 3>, kCameras> origins{};        // camera centres, world units
    std::array<std::array<double, 9>, kCameras> rotations{};      // row-major R: world ray = [x, y, 1] @ R
    std::array<std::array<double, 4>, kCameras> intrinsics{};     // fx, fy, cx, cy in pixels
};

Rig load_rig(const std::string& path);

// Detected joints of one camera frame, u,v in [0,1] as [left 21][right 21] interleaved (-1 or
// outside [0,1]: not detected), and the frame's capture -> arrival delay (s) -> event features
// [2][21][14] and validity, exactly as the simulator built them for training: u,v, the nominal
// camera centre, the unit pinhole ray through the pixel rotated to the world, the delay.
void make_features(const Rig& rig, int camera, const std::array<float, 84>& uv, double delay_s,
                   Features& features, Valid& valid);

}  // namespace hand_master
