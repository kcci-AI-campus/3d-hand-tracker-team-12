// Geometry of hand_tracking/lite_runtime.py: triangulation, camera correction, sample
// stamps, the query-time anchor and the encoder's joint features. Double precision.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "hand_lite/hand_lite.hpp"

namespace hand_lite::detail {

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<Vec3, 3>;  // rows

// hand_tracking/constants.py
constexpr double kTimeUnit = .1;
constexpr double kMissScale = 10.;
constexpr double kArrivalTolerance = 1e-6;
constexpr double kRepeatTolerance = 1e-5;
constexpr double kMinTimeStd = .005;
constexpr double kMaxExtrapolation = .25;
constexpr double kMinRaySin2 = 2e-3;
constexpr double kMinRayDepth = 1e-3;
constexpr double kRayResidualFloor = .01;
constexpr double kOutlierMargin = 3.;
constexpr double kMaskOff = -1e4;
constexpr double kStreamKeepMargin = .5;
constexpr double kArrivalMargin = .25;  // hand_tracking/config.py ARRIVAL_MARGIN_S

// Feature channels (hand_tracking/constants.py): u,v 0-1, origin 2-4, direction 5-7, delay 10.
constexpr int kOrigin = 2;
constexpr int kDirection = 5;
constexpr int kDelay = 10;

// Each camera's ray for every joint; index [camera * 42 + hand * 21 + joint].
struct Rays {
    std::array<Vec3, kCameras * kHandJoints> origin{};
    std::array<Vec3, kCameras * kHandJoints> direction{};
    std::array<std::uint8_t, kCameras * kHandJoints> mask{};
    std::array<double, kCameras> capture{};
};

struct Sample {
    std::array<Vec3, kHandJoints> point{};
    std::array<std::uint8_t, kHandJoints> ok{};
    std::array<double, kHandJoints> stamp{};  // mean capture time of the masked rays
};

struct Anchor {
    std::array<Vec3, kHandJoints> position{};
    // now, joint, hand, age (0.1 s), moving, single ray (only with SingleRays)
    std::array<std::array<double, 6>, kHandJoints> flags{};
};

// One event's corrected ray table, index [camera * 42 + joint], and each joint's remembered
// triangulation (lite_runtime.LiteRuntime._single_rays).
struct SingleRays {
    std::array<Vec3, kCameras * kHandJoints> origin{}, direction{};
    std::array<std::uint8_t, kCameras * kHandJoints> has{};
    std::array<double, kCameras> capture{};
    std::array<Vec3, kHandJoints> reference{};
    std::array<std::uint8_t, kHandJoints> known{};
};

Mat3 rodrigues(const double* rotation_vector);

// Least-squares point of the masked rays (geometry.triangulate without a residual gate).
bool triangulate(const Vec3* origins, const Vec3* directions, const std::uint8_t* mask, int rays, Vec3& point);

// Least-squares point dropping at most one outlier ray (geometry.robust_triangulate);
// used[c] tells which rays were kept.
bool robust_triangulate(const Vec3* origins, const Vec3* directions, const std::uint8_t* mask, int rays,
                        double outlier_ratio, Vec3& point, std::uint8_t* used);

// Triangulate every joint, after the correction params [3][6] when given (nullptr: nominal);
// outlier_ratio > 0 drops one inconsistent ray (robust_triangulate).
Sample sample(const Rays& rays, const double* params, double outlier_ratio = 0.);

// The rays corrected by params [3][6]; fills origin, direction, has and capture of out.
void corrected_rays(const Rays& rays, const double* params, SingleRays& out);

// Anchor at query from slot samples in arrival order (all arrived by the query); with
// single, joints the latest sample did not triangulate are anchored on the ray closest to
// their reference point.
Anchor anchor_at(const std::vector<const Sample*>& samples, double query, double lookback_s, double hold_s,
                 const SingleRays* single = nullptr);

// raw [42][11] (invalid joints zeroed), valid [42], nominal sample -> encoder input [42][14].
void joint_features(const double* raw, const std::uint8_t* valid, const Sample& nominal, float* out);

}  // namespace hand_lite::detail
