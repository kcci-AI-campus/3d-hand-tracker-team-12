#include "geometry.hpp"

#include <algorithm>
#include <cmath>

namespace hand_lite::detail {
namespace {

Vec3 add(const Vec3& a, const Vec3& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
Vec3 sub(const Vec3& a, const Vec3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
Vec3 scale(const Vec3& a, double s) { return {a[0] * s, a[1] * s, a[2] * s}; }
double dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double norm(const Vec3& a) { return std::sqrt(dot(a, a)); }
bool finite(const Vec3& a) { return std::isfinite(a[0]) && std::isfinite(a[1]) && std::isfinite(a[2]); }
Vec3 rotate(const Mat3& m, const Vec3& v) { return {dot(m[0], v), dot(m[1], v), dot(m[2], v)}; }

}  // namespace

Mat3 rodrigues(const double* w) {
    const double theta = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
    const bool small = theta < 1e-4;
    const double safe = small ? 1. : theta;
    const double a = small ? 1. - theta * theta / 6. : std::sin(safe) / safe;
    const double half = safe / 2.;
    const double sinc_half = std::sin(half) / half;
    const double b = small ? .5 - theta * theta / 24. : .5 * sinc_half * sinc_half;
    const Mat3 k = {{{0., -w[2], w[1]}, {w[2], 0., -w[0]}, {-w[1], w[0], 0.}}};
    Mat3 r{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double kk = 0.;
            for (int l = 0; l < 3; ++l) kk += k[i][l] * k[l][j];
            r[i][j] = (i == j ? 1. : 0.) + a * k[i][j] + b * kk;
        }
    }
    return r;
}

bool triangulate(const Vec3* origins, const Vec3* directions, const std::uint8_t* mask, int rays, Vec3& point) {
    std::array<Vec3, kCameras> o{}, d{};
    std::array<bool, kCameras> m{};
    int count = 0;
    for (int c = 0; c < rays; ++c) {
        m[c] = mask[c] && finite(origins[c]) && finite(directions[c]) && norm(directions[c]) > 1e-8;
        if (m[c]) {
            o[c] = origins[c];
            d[c] = scale(directions[c], 1. / std::max(norm(directions[c]), 1e-8));
            ++count;
        }
    }
    // sum over rays of (I - d d^T) and (I - d d^T) o
    Mat3 a{};
    Vec3 rhs{};
    for (int i = 0; i < 3; ++i) a[i][i] = count;
    for (int c = 0; c < rays; ++c) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) a[i][j] -= d[c][i] * d[c][j];
        rhs = add(rhs, sub(o[c], scale(d[c], dot(d[c], o[c]))));
    }
    double sin2 = 0.;
    for (int i = 0; i < rays; ++i)
        for (int j = i + 1; j < rays; ++j) sin2 = std::max(sin2, dot(cross(d[i], d[j]), cross(d[i], d[j])));
    bool valid = count >= 2 && sin2 > kMinRaySin2;
    if (!valid) a = {{{1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}}};
    // Adjugate (Cramer's rule): exact for 3x3, as geometry.solve3.
    const Vec3 c0 = cross(a[1], a[2]), c1 = cross(a[2], a[0]), c2 = cross(a[0], a[1]);
    const double det = dot(a[0], c0);
    point = scale(add(add(scale(c0, rhs[0]), scale(c1, rhs[1])), scale(c2, rhs[2])), 1. / det);
    for (int c = 0; c < rays; ++c)
        if (m[c] && !(dot(sub(point, o[c]), d[c]) > 0.)) valid = false;  // behind a camera: outlier
    if (!valid) point = {0., 0., 0.};
    return valid;
}

namespace {

// Angular residual (miss distance / depth) of a ray to point.
double ray_residual(const Vec3& point, const Vec3& origin, const Vec3& direction) {
    const Vec3 d = scale(direction, 1. / std::max(norm(direction), 1e-8));
    const Vec3 relative = sub(point, origin);
    const double depth = dot(relative, d);
    return norm(sub(relative, scale(d, depth))) / std::max(depth, 1e-6);
}

}  // namespace

bool robust_triangulate(const Vec3* origins, const Vec3* directions, const std::uint8_t* mask, int rays,
                        double outlier_ratio, Vec3& point, std::uint8_t* used) {
    for (int c = 0; c < rays; ++c) used[c] = mask[c];
    bool valid = triangulate(origins, directions, mask, rays, point);
    int count = 0;
    for (int c = 0; c < rays; ++c) count += mask[c] != 0;
    if (!(outlier_ratio > 0.) || count < 3) return valid;
    double best = 0., second = 0.;
    int dropped = -1;
    Vec3 best_point{};
    bool best_valid = false;
    for (int k = 0; k < rays; ++k) {
        if (!mask[k]) continue;
        std::array<std::uint8_t, kCameras> keep{};
        for (int c = 0; c < rays; ++c) keep[c] = mask[c] && c != k;
        Vec3 loo{};
        if (!triangulate(origins, directions, keep.data(), rays, loo)) continue;
        double spread = 0.;
        for (int c = 0; c < rays; ++c)
            if (keep[c]) spread = std::max(spread, ray_residual(loo, origins[c], directions[c]));
        const double ratio = ray_residual(loo, origins[k], directions[k]) / std::max(spread, kRayResidualFloor);
        if (ratio > best) {
            second = best;
            best = ratio;
            dropped = k;
            best_point = loo;
            best_valid = true;
        } else {
            second = std::max(second, ratio);
        }
    }
    if (!(best > outlier_ratio)) return valid;
    if (best < kOutlierMargin * second) {  // two suspects: ambiguous, no sample
        point = {0., 0., 0.};
        return false;
    }
    used[dropped] = 0;
    point = best_point;
    return best_valid;
}

Sample sample(const Rays& rays, const double* params, double outlier_ratio) {
    std::array<Mat3, kCameras> rotation{};
    if (params)
        for (int c = 0; c < kCameras; ++c) rotation[c] = rodrigues(params + c * kCalibrationParams);
    Sample out;
    for (int j = 0; j < kHandJoints; ++j) {
        std::array<Vec3, kCameras> o{}, d{};
        std::array<std::uint8_t, kCameras> m{}, used{};
        double weight = 0., stamp = 0.;
        for (int c = 0; c < kCameras; ++c) {
            const int index = c * kHandJoints + j;
            o[c] = rays.origin[index];
            d[c] = rays.direction[index];
            if (params) {
                const double* shift = params + c * kCalibrationParams + 3;
                o[c] = add(o[c], {shift[0], shift[1], shift[2]});
                d[c] = rotate(rotation[c], d[c]);
            }
            m[c] = rays.mask[index];
        }
        out.ok[j] = robust_triangulate(o.data(), d.data(), m.data(), kCameras, outlier_ratio, out.point[j], used.data());
        for (int c = 0; c < kCameras; ++c) {
            weight += used[c];
            stamp += rays.capture[c] * used[c];
        }
        out.stamp[j] = stamp / std::max(weight, 1.);
    }
    return out;
}

void corrected_rays(const Rays& rays, const double* params, SingleRays& out) {
    for (int c = 0; c < kCameras; ++c) {
        const Mat3 rotation = params ? rodrigues(params + c * kCalibrationParams) : Mat3{};
        out.capture[c] = rays.capture[c];
        for (int j = 0; j < kHandJoints; ++j) {
            const int index = c * kHandJoints + j;
            out.has[index] = rays.mask[index];
            out.origin[index] = rays.origin[index];
            out.direction[index] = rays.direction[index];
            if (params) {
                const double* shift = params + c * kCalibrationParams + 3;
                out.origin[index] = add(out.origin[index], {shift[0], shift[1], shift[2]});
                out.direction[index] = rotate(rotation, out.direction[index]);
            }
        }
    }
}

Anchor anchor_at(const std::vector<const Sample*>& samples, double query, double lookback_s, double hold_s,
                 const SingleRays* single) {
    Anchor anchor;
    const int count = static_cast<int>(samples.size());
    if (count == 0) return anchor;  // no slot in the span: origin, all flags off
    std::array<std::uint8_t, kHandJoints> joint{}, moving{};
    for (int j = 0; j < kHandJoints; ++j) {
        const auto usable = [&](int s) {
            return samples[s]->ok[j] && -(samples[s]->stamp[j] - query) <= hold_s;
        };
        int latest = -1;
        for (int s = 0; s < count; ++s)
            if (usable(s)) latest = s;
        joint[j] = latest >= 0;
        const Sample& picked = *samples[std::max(latest, 0)];
        anchor.position[j] = picked.point[j];
        const double age = joint[j] ? query - picked.stamp[j] : 0.;
        if (lookback_s > 0) {
            // A sample repeated over events (a joint the newest event did not see) counts once.
            std::vector<double> w(count, 0.);
            for (int s = 0; s < count; ++s) {
                const double tau = samples[s]->stamp[j] - query;
                if (!(usable(s) && -tau <= lookback_s)) continue;
                int same = 0;
                for (int u = 0; u < count; ++u) {
                    const double tau_u = samples[u]->stamp[j] - query;
                    const bool use_u = usable(u) && -tau_u <= lookback_s;
                    same += use_u && samples[s]->ok[j] && samples[u]->ok[j] &&
                            std::abs(samples[s]->stamp[j] - samples[u]->stamp[j]) <= kRepeatTolerance;
                }
                w[s] = 1. / std::max(same, 1);
            }
            double total = 0., mean_tau = 0.;
            Vec3 mean_point{};
            for (int s = 0; s < count; ++s) {
                total += w[s];
                mean_tau += w[s] * (samples[s]->stamp[j] - query);
                mean_point = add(mean_point, scale(samples[s]->point[j], w[s]));
            }
            mean_tau /= std::max(total, 1e-12);
            mean_point = scale(mean_point, 1. / std::max(total, 1e-12));
            double var_tau = 0.;
            Vec3 slope{};
            for (int s = 0; s < count; ++s) {
                const double centred = samples[s]->stamp[j] - query - mean_tau;
                var_tau += w[s] * centred * centred;
                slope = add(slope, scale(sub(samples[s]->point[j], mean_point), w[s] * centred));
            }
            slope = scale(slope, 1. / std::max(var_tau, 1e-12));
            moving[j] = total >= 2 - 1e-6 && var_tau > kMinTimeStd * kMinTimeStd * total;
            if (moving[j])
                anchor.position[j] = add(mean_point, scale(slope, std::clamp(-mean_tau, 0., kMaxExtrapolation)));
        }
        const Sample& last = *samples[count - 1];
        anchor.flags[j] = {static_cast<double>(last.ok[j]), static_cast<double>(joint[j]), 0., age / kTimeUnit,
                           static_cast<double>(moving[j]), 0.};
    }
    if (single) {
        // Depth reference: the joint's anchor, else its remembered triangulation, else the
        // mean reference of its hand (events.single_ray_anchor).
        std::array<Vec3, kHandJoints> reference = single->reference;
        std::array<std::uint8_t, kHandJoints> known{};
        for (int j = 0; j < kHandJoints; ++j) {
            if (joint[j]) reference[j] = anchor.position[j];
            known[j] = joint[j] || single->known[j];
        }
        for (int h = 0; h < kHands; ++h) {
            Vec3 mean{};
            int seen = 0;
            for (int j = h * kJoints; j < (h + 1) * kJoints; ++j)
                if (known[j]) {
                    mean = add(mean, reference[j]);
                    ++seen;
                }
            mean = scale(mean, 1. / std::max(seen, 1));
            for (int j = h * kJoints; j < (h + 1) * kJoints; ++j) {
                if (!known[j]) reference[j] = mean;
                known[j] = seen > 0;
            }
        }
        for (int j = 0; j < kHandJoints; ++j) {
            // The joint's ray with the smallest angular residual to its reference (first on ties).
            int best = -1;
            double best_residual = 0., best_depth = 0.;
            Vec3 best_d{};
            for (int c = 0; c < kCameras; ++c) {
                const int index = c * kHandJoints + j;
                if (!single->has[index]) continue;
                const Vec3 d = scale(single->direction[index], 1. / std::max(norm(single->direction[index]), 1e-12));
                const Vec3 relative = sub(reference[j], single->origin[index]);
                const double along = dot(relative, d);
                const double residual = norm(sub(relative, scale(d, along))) / std::max(along, 1e-6);
                if (best < 0 || residual < best_residual) {
                    best = c;
                    best_residual = residual;
                    best_depth = along;
                    best_d = d;
                }
            }
            if (!(best >= 0 && !samples[count - 1]->ok[j] && known[j])) continue;
            anchor.position[j] = add(single->origin[best * kHandJoints + j], scale(best_d, std::max(best_depth, kMinRayDepth)));
            joint[j] = 1;
            moving[j] = 0;
            anchor.flags[j][1] = 1.;
            anchor.flags[j][3] = (query - single->capture[best]) / kTimeUnit;
            anchor.flags[j][4] = 0.;
            anchor.flags[j][5] = 1.;
        }
    }
    // Joints without a sample use the mean anchor of the same hand, else the origin.
    for (int h = 0; h < kHands; ++h) {
        Vec3 mean{};
        int seen = 0;
        for (int j = h * kJoints; j < (h + 1) * kJoints; ++j)
            if (joint[j]) {
                mean = add(mean, anchor.position[j]);
                ++seen;
            }
        mean = scale(mean, 1. / std::max(seen, 1));
        for (int j = h * kJoints; j < (h + 1) * kJoints; ++j) {
            if (!joint[j]) anchor.position[j] = mean;
            anchor.flags[j][2] = seen > 0;
        }
    }
    return anchor;
}

void joint_features(const double* raw, const std::uint8_t* valid, const Sample& nominal, float* out) {
    for (int j = 0; j < kHandJoints; ++j) {
        const double* r = raw + j * kModelChannels;
        float* x = out + j * kJointFeatures;
        if (!valid[j]) {
            std::fill(x, x + kJointFeatures, 0.f);
            continue;
        }
        const Vec3 origin = {r[kOrigin], r[kOrigin + 1], r[kOrigin + 2]};
        const Vec3 direction = {r[kDirection], r[kDirection + 1], r[kDirection + 2]};
        const Vec3 relative = sub(nominal.point[j], origin);
        Vec3 miss = sub(relative, scale(direction, dot(relative, direction)));
        miss = nominal.ok[j] ? scale(miss, kMissScale) : Vec3{0., 0., 0.};
        const double values[kJointFeatures] = {r[0] * 2 - 1, r[1] * 2 - 1, origin[0], origin[1], origin[2],
                                               direction[0], direction[1], direction[2], r[kDelay] / kTimeUnit,
                                               miss[0], miss[1], miss[2], 1., nominal.ok[j] ? 1. : 0.};
        for (int i = 0; i < kJointFeatures; ++i) x[i] = static_cast<float>(values[i]);
    }
}

}  // namespace hand_lite::detail
