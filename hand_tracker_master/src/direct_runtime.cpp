#include "direct_runtime.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "json.hpp"

namespace hand_master {

Config load_config(const std::string& export_directory) {
    const std::string path = export_directory + "/direct.json";
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open " + path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    const std::string text = buffer.str();
    if (json::string(text, "architecture") != "direct")
        throw std::runtime_error(export_directory + " is not a HandDirect export (architecture \"direct\")");
    Config config;
    config.dim = static_cast<int>(json::number(text, "dim"));
    config.slots_per_camera = static_cast<int>(json::number(text, "slots_per_camera"));
    config.event_span_s = json::number(text, "event_span_s");
    if (config.dim < 1 || config.slots_per_camera < 1 || !(config.event_span_s > 0))
        throw std::runtime_error("Invalid HandDirect config in " + path);
    return config;
}

std::vector<float> encoder_input(const Features& features, const Valid& valid) {
    // Per joint: u,v in [-1,1], ray origin (3), ray direction (3), capture->arrival delay (0.1 s
    // units), detected; an undetected joint is all zero (networks.direct_features).
    std::vector<float> x(kHandJoints * kEncoderChannels, 0.f);
    for (int j = 0; j < kHandJoints; ++j) {
        if (!valid[j]) continue;
        const float* f = &features[j * kFeatureChannels];
        float* out = &x[j * kEncoderChannels];
        out[0] = f[0] * 2 - 1;
        out[1] = f[1] * 2 - 1;
        for (int k = 0; k < 6; ++k) out[2 + k] = f[2 + k];  // origin 2..4, direction 5..7
        out[8] = static_cast<float>(f[10] / kTimeUnitS);
        out[9] = 1.f;
    }
    return x;
}

Runtime::Runtime(Config config, std::shared_ptr<Graphs> graphs) : config_(config), graphs_(std::move(graphs)) {
    if (!graphs_) throw std::invalid_argument("Runtime needs graphs");
}

bool Runtime::push(int camera, const Features& features, const Valid& valid, double capture, double arrival) {
    if (camera < 0 || camera >= kCameras) throw std::invalid_argument("camera must be 0..2");
    if (has_arrival_ && arrival < last_arrival_) throw std::logic_error("Events must be pushed in arrival order");
    last_arrival_ = arrival;
    has_arrival_ = true;
    if (has_capture_[camera] && capture <= newest_capture_[camera]) return false;
    newest_capture_[camera] = capture;
    has_capture_[camera] = true;
    Event event{camera, capture, arrival, {}, {}};
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

Pose Runtime::query(double time) {
    if (!has_arrival_) throw std::logic_error("No events yet");
    if (time < last_arrival_) throw std::logic_error("Query before the latest arrival");
    // Each camera's newest slots_per_camera events arrived by time and captured within the span,
    // in arrival order; padding first (lite_runtime.select_slot_events).
    const int per = config_.slots_per_camera;
    std::array<std::vector<std::size_t>, kCameras> chosen;
    for (std::size_t i = events_.size(); i-- > 0;) {
        const Event& e = events_[i];
        if (e.arrival <= time + kArrivalToleranceS && e.capture >= time - config_.event_span_s &&
            static_cast<int>(chosen[e.camera].size()) < per)
            chosen[e.camera].push_back(i);
    }
    std::vector<std::size_t> ordered;
    for (const auto& indices : chosen) ordered.insert(ordered.end(), indices.begin(), indices.end());
    std::sort(ordered.begin(), ordered.end());
    const int slots = config_.slots(), tokens_per = config_.event_tokens, dim = config_.dim;
    const int padding = slots - static_cast<int>(ordered.size());
    std::vector<float> tokens(static_cast<std::size_t>(slots) * tokens_per * dim, 0.f);
    std::vector<float> valid(static_cast<std::size_t>(slots) * tokens_per, 0.f), ages(valid.size(), 0.f);
    seen_.fill(0);
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
            if (e.detected[h] >= kSeenMinJoints) seen_camera[h][e.camera] = true;
    }
    for (int h = 0; h < kHands; ++h)
        seen_[h] = static_cast<int>(std::count(seen_camera[h].begin(), seen_camera[h].end(), true));
    const auto joints = graphs_->query(tokens, valid, ages);
    if (joints.size() != static_cast<std::size_t>(kHandJoints * 3)) throw std::runtime_error("Query output is not [42][3]");
    Pose pose;
    std::copy(joints.begin(), joints.end(), pose.begin());
    return pose;
}

}  // namespace hand_master
