#include "rig.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "json.hpp"

namespace hand_master {

Rig load_rig(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open rig file " + path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    const std::string text = buffer.str();
    Rig rig;
    rig.width = static_cast<int>(json::number(text, "width"));
    rig.height = static_cast<int>(json::number(text, "height"));
    rig.world_unit_cm = json::number(text, "world_unit_cm");
    const auto origins = json::numbers(text, "origins");
    const auto rotations = json::numbers(text, "rotations");
    const auto intrinsics = json::numbers(text, "intrinsics");
    if (origins.size() != 9 || rotations.size() != 27 || intrinsics.size() != 12 || rig.width < 1 || rig.height < 1)
        throw std::runtime_error("rig.json needs 3 origins [3], rotations [3][3] and intrinsics [4]");
    for (int c = 0; c < kCameras; ++c) {
        for (int k = 0; k < 3; ++k) rig.origins[c][k] = origins[c * 3 + k];
        for (int k = 0; k < 9; ++k) rig.rotations[c][k] = rotations[c * 9 + k];
        for (int k = 0; k < 4; ++k) rig.intrinsics[c][k] = intrinsics[c * 4 + k];
    }
    return rig;
}

void make_features(const Rig& rig, int camera, const std::array<float, 84>& uv, double delay_s,
                   Features& features, Valid& valid) {
    if (camera < 0 || camera >= kCameras) throw std::invalid_argument("camera must be 0..2");
    features.fill(0.f);
    valid.fill(0);
    const auto& r = rig.rotations[camera];
    const auto& k = rig.intrinsics[camera];
    for (int j = 0; j < kHandJoints; ++j) {
        const float u = uv[j * 2], v = uv[j * 2 + 1];
        if (!(u >= 0.f && u <= 1.f && v >= 0.f && v <= 1.f)) continue;  // -1, NaN or outside
        // gigahands_sim: d = [(px - cx)/fx, (py - cy)/fy, 1] @ R, normalised; px = u * width.
        const double x = (u * rig.width - k[2]) / k[0], y = (v * rig.height - k[3]) / k[1];
        double d[3];
        for (int i = 0; i < 3; ++i) d[i] = x * r[0 * 3 + i] + y * r[1 * 3 + i] + r[2 * 3 + i];
        const double norm = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        float* f = &features[j * kFeatureChannels];
        f[0] = u;
        f[1] = v;
        for (int i = 0; i < 3; ++i) {
            f[2 + i] = static_cast<float>(rig.origins[camera][i]);
            f[5 + i] = static_cast<float>(d[i] / norm);
        }
        // Channels 8, 9 (times relative to a query) are not model inputs; 11..13 are indices.
        f[10] = static_cast<float>(delay_s);
        f[11] = static_cast<float>(camera);
        f[12] = static_cast<float>(j / kJoints);
        f[13] = static_cast<float>(j % kJoints);
        valid[j] = 1;
    }
}

}  // namespace hand_master
