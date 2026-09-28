#include "litev3_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "json.hpp"

namespace hand_master {

namespace {

using Vec3 = std::array<double, 3>;
constexpr int kUV = 0, kOrigin = 2, kDirection = 5, kDelay = 10;

double dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
double norm(const Vec3& a) { return std::sqrt(dot(a, a)); }
Vec3 unit(const Vec3& a) {
    const double n = std::max(norm(a), 1e-8);
    return {a[0] / n, a[1] / n, a[2] / n};
}

// One camera's line fit for one joint (runtime.fitted_rays).
struct Ray {
    Vec3 origin{}, direction{}, newest_direction{};
    bool has = false, fit = false;
    double count = 0., age = 0., oldest = 0., rms = 0., residual = 0.;
    double uv[2] = {0., 0.};
};
using JointRays = std::array<Ray, kCameras>;              // per camera
using Rays = std::array<JointRays, kHandJoints>;          // per joint

// Per camera and joint, the line fit over the camera's events arrived by time and captured within
// span_s before time - offset_s, read at time - offset_s.
Rays fitted_rays(const std::deque<LiteV3Runtime::Event>& events, double time, double span_s, double offset_s) {
    Rays rays{};
    const double reference = time - offset_s;
    std::vector<const LiteV3Runtime::Event*> usable;
    for (int camera = 0; camera < kCameras; ++camera) {
        usable.clear();
        for (const auto& e : events)
            if (e.camera == camera && e.arrival <= time + kArrivalToleranceS && reference - span_s <= e.capture &&
                e.capture <= reference + kArrivalToleranceS)
                usable.push_back(&e);
        if (usable.empty()) continue;
        for (int j = 0; j < kHandJoints; ++j) {
            double s0 = 0., s1 = 0., s2 = 0., sdd = 0.;
            Vec3 sd{}, sdt{}, so{};
            double newest_tau = -kFar, oldest_tau = kFar;
            std::size_t newest_index = 0;
            bool seen = false;
            for (std::size_t n = 0; n < usable.size(); ++n) {
                const auto& e = *usable[n];
                if (!e.valid[j]) continue;
                const double tau = e.capture - reference, mt = tau;  // m = 1
                const double* raw = &e.raw[j * kLiteRawChannels];
                s0 += 1.;
                s1 += mt;
                s2 += mt * tau;
                for (int k = 0; k < 3; ++k) {
                    const double d = raw[kDirection + k];
                    sd[k] += d;
                    sdt[k] += mt * d;
                    so[k] += raw[kOrigin + k];
                }
                sdd += raw[kDirection] * raw[kDirection] + raw[kDirection + 1] * raw[kDirection + 1] +
                       raw[kDirection + 2] * raw[kDirection + 2];
                newest_tau = std::max(newest_tau, tau);
                oldest_tau = std::min(oldest_tau, tau);
                newest_index = n;
                seen = true;
            }
            const double spread = s0 * s2 - s1 * s1;
            const bool fitted = s0 >= 2 && spread > kMinTimeStdS * kMinTimeStdS * s0 * s0;
            Vec3 slope{}, value{};
            for (int k = 0; k < 3; ++k) {
                slope[k] = fitted ? (s0 * sdt[k] - s1 * sd[k]) / std::max(spread, 1e-12) : 0.;
                value[k] = (sd[k] - slope[k] * s1) / std::max(s0, 1.);
            }
            const double squared = std::max(sdd - 2 * dot(value, sd) - 2 * dot(slope, sdt) + dot(value, value) * s0 +
                                                2 * dot(value, slope) * s1 + dot(slope, slope) * s2, 0.);
            const double scale = std::max(norm(value), 1e-8);
            // Undetected everywhere: numpy picks event 0, whose undetected joint is all zero.
            const double* newest = &usable[seen ? newest_index : 0]->raw[j * kLiteRawChannels];
            const Vec3 newest_dir{newest[kDirection], newest[kDirection + 1], newest[kDirection + 2]};
            const double clipped = std::max(newest_tau, -span_s);
            Vec3 miss{};
            for (int k = 0; k < 3; ++k) miss[k] = newest_dir[k] - value[k] - slope[k] * clipped;
            Ray& ray = rays[j][camera];
            for (int k = 0; k < 3; ++k) ray.origin[k] = so[k] / std::max(s0, 1.);
            ray.direction = unit(value);
            ray.has = seen;
            ray.fit = fitted;
            ray.count = s0;
            ray.age = seen ? -newest_tau : 0.;
            ray.oldest = seen ? -oldest_tau : 0.;
            ray.uv[0] = newest[kUV];
            ray.uv[1] = newest[kUV + 1];
            ray.newest_direction = unit(newest_dir);
            ray.rms = seen ? std::sqrt(squared / std::max(s0, 1.)) / scale : 0.;
            ray.residual = seen ? norm(miss) / scale : 0.;
        }
    }
    return rays;
}

// Solve a x = b (3x3) by Gaussian elimination with partial pivoting (LAPACK gesv's method).
Vec3 solve3(std::array<std::array<double, 3>, 3> a, Vec3 b) {
    for (int col = 0; col < 3; ++col) {
        int pivot = col;
        for (int r = col + 1; r < 3; ++r)
            if (std::abs(a[r][col]) > std::abs(a[pivot][col])) pivot = r;
        std::swap(a[col], a[pivot]);
        std::swap(b[col], b[pivot]);
        for (int r = col + 1; r < 3; ++r) {
            const double f = a[r][col] / a[col][col];
            for (int k = col; k < 3; ++k) a[r][k] -= f * a[col][k];
            b[r] -= f * b[col];
        }
    }
    Vec3 x{};
    for (int r = 2; r >= 0; --r) {
        double s = b[r];
        for (int k = r + 1; k < 3; ++k) s -= a[r][k] * x[k];
        x[r] = s / a[r][r];
    }
    return x;
}

// Least-squares point of the cameras' rays that have one (runtime.triangulate).
bool triangulate(const JointRays& rays, Vec3& point) {
    std::array<Vec3, kCameras> d{}, o{};
    std::array<bool, kCameras> mask{};
    int count = 0;
    for (int c = 0; c < kCameras; ++c) {
        const Ray& r = rays[c];
        mask[c] = r.has && std::isfinite(r.origin[0] + r.origin[1] + r.origin[2]) &&
                  std::isfinite(r.direction[0] + r.direction[1] + r.direction[2]) && norm(r.direction) > 1e-8;
        if (!mask[c]) continue;
        d[c] = unit(r.direction);
        o[c] = r.origin;
        ++count;
    }
    std::array<std::array<double, 3>, 3> a{};
    Vec3 rhs{};
    for (int i = 0; i < 3; ++i) a[i][i] = count;
    for (int c = 0; c < kCameras; ++c) {
        if (!mask[c]) continue;
        const double doo = dot(d[c], o[c]);
        for (int i = 0; i < 3; ++i) {
            for (int k = 0; k < 3; ++k) a[i][k] -= d[c][i] * d[c][k];
            rhs[i] += o[c][i] - d[c][i] * doo;
        }
    }
    double sin2 = 0.;
    for (int i = 0; i < kCameras; ++i)
        for (int j = i + 1; j < kCameras; ++j) {
            const Vec3 cross{d[i][1] * d[j][2] - d[i][2] * d[j][1], d[i][2] * d[j][0] - d[i][0] * d[j][2],
                             d[i][0] * d[j][1] - d[i][1] * d[j][0]};
            sin2 = std::max(sin2, dot(cross, cross));
        }
    bool valid = count >= 2 && sin2 > kMinRaySin2;
    if (!valid) { point = {0., 0., 0.}; return false; }
    point = solve3(a, rhs);
    for (int c = 0; c < kCameras; ++c) {
        if (!mask[c]) continue;
        const Vec3 rel{point[0] - o[c][0], point[1] - o[c][1], point[2] - o[c][2]};
        if (!(dot(rel, d[c]) > 0)) valid = false;
    }
    if (!valid) point = {0., 0., 0.};
    return valid;
}

// Angular residual of a ray to a point (runtime.ray_residuals).
double ray_residual(const Vec3& point, const Vec3& origin, const Vec3& direction) {
    const Vec3 d = unit(direction);
    const Vec3 rel{point[0] - origin[0], point[1] - origin[1], point[2] - origin[2]};
    const double depth = dot(rel, d);
    const Vec3 miss{rel[0] - depth * d[0], rel[1] - depth * d[1], rel[2] - depth * d[2]};
    return norm(miss) / std::max(depth, 1e-6);
}

}  // namespace

int LiteConfig::prior_steps() const { return static_cast<int>(std::lround(prior_span_s / prior_step_s)); }

double LiteConfig::context_s() const {
    return std::max(fit_span_s + prior_steps() * prior_step_s, event_span_s) + kArrivalMarginS;
}

LiteConfig load_lite_config(const std::string& export_directory) {
    const std::string path = export_directory + "/litev3.json";
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open " + path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    const std::string text = buffer.str();
    if (static_cast<int>(json::number(text, "export_format")) != 4)
        throw std::runtime_error(export_directory + " is not a HandLiteV3 export of format 4; re-run training.export");
    LiteConfig config;
    config.dim = static_cast<int>(json::number(text, "dim"));
    config.slots_per_camera = static_cast<int>(json::number(text, "slots_per_camera"));
    config.fit_span_s = json::number(text, "fit_span_s");
    config.prior_span_s = json::number(text, "prior_span_s");
    config.prior_step_s = json::number(text, "prior_step_s");
    config.event_span_s = json::number(text, "event_span_s");
    config.error_estimate = json::boolean(text, "error_estimate");
    config.presence = json::boolean(text, "presence");
    if (config.dim < 1 || config.slots_per_camera < 1 || !(config.fit_span_s > 0) || !(config.prior_step_s > 0) ||
        config.prior_span_s < 0 || !(config.event_span_s > 0))
        throw std::runtime_error("Invalid HandLiteV3 config in " + path);
    return config;
}

LiteV3Runtime::LiteV3Runtime(LiteConfig config, std::shared_ptr<LiteGraphs> graphs)
    : config_(config), graphs_(std::move(graphs)) {
    if (!graphs_) throw std::invalid_argument("LiteV3Runtime needs graphs");
}

std::optional<std::array<float, kHands>> LiteV3Runtime::in_view_probability() const {
    if (!has_in_view_) return std::nullopt;
    return in_view_;
}

bool LiteV3Runtime::push(int camera, const Features& features, const Valid& valid, double capture, double arrival) {
    if (camera < 0 || camera >= kCameras) throw std::invalid_argument("camera must be 0..2");
    if (has_arrival_ && arrival < last_arrival_) throw std::logic_error("Events must be pushed in arrival order");
    last_arrival_ = arrival;
    has_arrival_ = true;
    if (has_capture_[camera] && capture <= newest_capture_[camera]) return false;
    newest_capture_[camera] = capture;
    has_capture_[camera] = true;
    Event event{camera, capture, arrival, {}, valid, {}, {}};
    for (int j = 0; j < kHandJoints; ++j)
        for (int k = 0; k < kLiteRawChannels; ++k)
            event.raw[j * kLiteRawChannels + k] = valid[j] ? static_cast<double>(features[j * kFeatureChannels + k]) : 0.;
    static const std::array<float, 3> one_hot[kCameras] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    event.tokens = graphs_->encoder(encoder_input(features, valid), one_hot[camera]);
    if (event.tokens.size() != static_cast<std::size_t>(config_.event_tokens * config_.dim))
        throw std::runtime_error("Encoder output size differs from the config");
    for (int h = 0; h < kHands; ++h)
        event.detected[h] = static_cast<std::uint8_t>(std::count(valid.begin() + h * kJoints, valid.begin() + (h + 1) * kJoints, 1));
    events_.push_back(std::move(event));
    const double keep = config_.context_s() + kStreamKeepMarginS;
    while (!events_.empty() && events_.front().capture < arrival - keep) events_.pop_front();
    return true;
}

Pose LiteV3Runtime::query(double time) {
    if (!has_arrival_) throw std::logic_error("No events yet");
    if (time < last_arrival_) throw std::logic_error("Query before the latest arrival");
    const LiteConfig& c = config_;

    // Anchors (runtime.anchor_features).
    const Rays rays = fitted_rays(events_, time, c.fit_span_s, 0.);
    std::array<Vec3, kHandJoints> point{}, prior{};
    std::array<bool, kHandJoints> ok{}, found{};
    std::array<double, kHandJoints> prior_age{};
    for (int j = 0; j < kHandJoints; ++j) ok[j] = triangulate(rays[j], point[j]);
    for (int k = 1; k <= c.prior_steps(); ++k) {  // the joint's own newest triangulation before time
        const double offset = k * c.prior_step_s;
        const Rays past = fitted_rays(events_, time, c.fit_span_s, offset);
        for (int j = 0; j < kHandJoints; ++j) {
            Vec3 p;
            const bool success = triangulate(past[j], p);
            if (success && !found[j]) {
                double newest = kFar;
                for (int cam = 0; cam < kCameras; ++cam)
                    if (past[j][cam].has) newest = std::min(newest, past[j][cam].age);
                prior[j] = p;
                prior_age[j] = newest + offset;
            }
            found[j] = found[j] || success;
        }
    }
    std::array<Vec3, kHandJoints> anchor{};
    std::array<double, kHandJoints> observed{};
    std::vector<float> joints(static_cast<std::size_t>(kHandJoints) * kLiteJointFeatures, 0.f);
    for (int j = 0; j < kHandJoints; ++j) {
        if (prior_known_[j] && !found[j]) {  // the state fills in where the history search found none
            prior[j] = prior_point_[j];
            prior_age[j] = time - prior_capture_[j];
        }
        found[j] = found[j] || prior_known_[j];
        const JointRays& r = rays[j];
        double seen_age = kFar;
        bool seen = false;
        for (int cam = 0; cam < kCameras; ++cam)
            if (r[cam].has) { seen_age = std::min(seen_age, r[cam].age); seen = true; }
        const double age = found[j] ? std::min(prior_age[j], c.max_prior_age_s()) : 0.;
        int best = -1;  // numpy argmin: the first minimum (a camera without the joint counts as kFar)
        double best_miss = 0.;
        for (int cam = 0; cam < kCameras; ++cam) {
            const double miss = r[cam].has ? ray_residual(prior[j], r[cam].origin, r[cam].direction) : kFar;
            if (best < 0 || miss < best_miss) { best_miss = miss; best = cam; }
        }
        const Vec3& o = r[best].origin;
        const Vec3& d = r[best].direction;
        const Vec3 to_prior{prior[j][0] - o[0], prior[j][1] - o[1], prior[j][2] - o[2]};
        const double along = std::max(dot(to_prior, d), 0.);
        const bool on_ray = !ok[j] && found[j] && seen, at_prior = !ok[j] && found[j] && !seen;
        if (ok[j]) anchor[j] = point[j];
        else if (on_ray) anchor[j] = {o[0] + along * d[0], o[1] + along * d[1], o[2] + along * d[2]};
        else if (at_prior) anchor[j] = prior[j];
        anchor_kind_[j] = static_cast<std::uint8_t>(ok[j] ? kAnchorTriangulated : on_ray ? kAnchorRay : at_prior ? kAnchorPrior : kAnchorNone);
        observed[j] = ok[j] ? seen_age : 0.;
        float* f = &joints[static_cast<std::size_t>(j) * kLiteJointFeatures];
        for (int k = 0; k < 3; ++k) {
            f[k] = static_cast<float>(anchor[j][k]);
            f[3 + k] = static_cast<float>(found[j] ? anchor[j][k] - prior[j][k] : 0.);
        }
        f[6] = ok[j];
        f[7] = on_ray;
        f[8] = at_prior;
        f[9] = found[j];
        f[10] = static_cast<float>(age / kTimeUnitS);
        for (int cam = 0; cam < kCameras; ++cam) {
            const Ray& ray = r[cam];
            if (!ray.has) continue;
            float* g = f + 11 + cam * 15;
            const Vec3 rel{anchor[j][0] - ray.origin[0], anchor[j][1] - ray.origin[1], anchor[j][2] - ray.origin[2]};
            const double along_ray = dot(rel, ray.direction);
            g[0] = 1.f;
            g[1] = ray.fit;
            g[2] = static_cast<float>(ray.count / 4);
            g[3] = static_cast<float>(ray.age / kTimeUnitS);
            g[4] = static_cast<float>(ray.oldest / kTimeUnitS);
            g[5] = static_cast<float>(ray.uv[0] * 2 - 1);
            g[6] = static_cast<float>(ray.uv[1] * 2 - 1);
            for (int k = 0; k < 3; ++k) {
                g[7 + k] = static_cast<float>((rel[k] - ray.direction[k] * along_ray) * kMissScale);
                g[10 + k] = static_cast<float>((ray.direction[k] - ray.newest_direction[k]) * kMissScale);
            }
            g[13] = static_cast<float>(ray.rms / kResidualUnit);
            g[14] = static_cast<float>(ray.residual / kResidualUnit);
        }
    }
    // State update: every joint triangulated now.
    for (int j = 0; j < kHandJoints; ++j)
        if (anchor_kind_[j] == kAnchorTriangulated) {
            prior_point_[j] = anchor[j];
            prior_capture_[j] = time - observed[j];
            prior_known_[j] = true;
        }

    // Slots (each camera's newest events within event_span_s), as HandDirect.
    const int per = c.slots_per_camera;
    std::array<std::vector<std::size_t>, kCameras> chosen;
    for (std::size_t i = events_.size(); i-- > 0;) {
        const Event& e = events_[i];
        if (e.arrival <= time + kArrivalToleranceS && e.capture >= time - c.event_span_s &&
            static_cast<int>(chosen[e.camera].size()) < per)
            chosen[e.camera].push_back(i);
    }
    std::vector<std::size_t> ordered;
    for (const auto& indices : chosen) ordered.insert(ordered.end(), indices.begin(), indices.end());
    std::sort(ordered.begin(), ordered.end());
    const int slots = c.slots(), tokens_per = c.event_tokens, dim = c.dim;
    const int padding = slots - static_cast<int>(ordered.size());
    std::vector<float> tokens(static_cast<std::size_t>(slots) * tokens_per * dim, 0.f);
    std::vector<float> valid(static_cast<std::size_t>(slots) * tokens_per, 0.f), ages(valid.size(), 0.f);
    std::array<std::array<bool, kCameras>, kHands> seen_camera{};
    for (std::size_t k = 0; k < ordered.size(); ++k) {
        const Event& e = events_[ordered[k]];
        const std::size_t slot = padding + k;
        std::copy(e.tokens.begin(), e.tokens.end(), tokens.begin() + slot * tokens_per * dim);
        const float age = static_cast<float>(std::max(time - e.capture, 0.) / kTimeUnitS);
        for (int t = 0; t < tokens_per; ++t) {
            valid[slot * tokens_per + t] = 1.f;
            ages[slot * tokens_per + t] = age;
        }
        for (int h = 0; h < kHands; ++h)
            if (e.detected[h] >= Runtime::kSeenMinJoints) seen_camera[h][e.camera] = true;
    }
    for (int h = 0; h < kHands; ++h)
        seen_[h] = static_cast<int>(std::count(seen_camera[h].begin(), seen_camera[h].end(), true));

    const auto output = graphs_->corrector(joints, tokens, valid, ages);
    const int columns = c.output_columns();
    if (output.size() != static_cast<std::size_t>(kHandJoints * columns))
        throw std::runtime_error("Corrector output is not [42][" + std::to_string(columns) + "]");
    Pose pose;
    for (int j = 0; j < kHandJoints; ++j) {
        for (int k = 0; k < 3; ++k) pose[j * 3 + k] = static_cast<float>(anchor[j][k] + output[j * columns + k]);
        error_mm_[j] = c.error_estimate ? static_cast<float>(std::expm1(double(output[j * columns + 3]))) : 0.f;
    }
    if (c.presence) {
        const int column = c.error_estimate ? 4 : 3;
        for (int h = 0; h < kHands; ++h)
            in_view_[h] = static_cast<float>(1. / (1. + std::exp(-double(output[h * kJoints * columns + column]))));
        has_in_view_ = true;
    }
    return pose;
}

}  // namespace hand_master
