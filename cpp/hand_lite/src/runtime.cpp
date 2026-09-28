#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "geometry.hpp"
#include "hand_lite/hand_lite.hpp"

namespace hand_lite {

using detail::kArrivalTolerance;
using detail::kMaskOff;
using detail::kTimeUnit;

double Config::context_s() const {
    const double memory = anchor_ray_depth ? anchor_depth_memory_s : 0.;
    return std::max({event_span_s, anchor_hold_s, anchor_lookback_s, memory}) + detail::kArrivalMargin +
           sample_max_age_s;
}

namespace {

// lite.json is flat enough that each key appears once: read "key": value.
std::string json_value(const std::string& text, const std::string& key) {
    const std::string quoted = "\"" + key + "\"";
    const auto at = text.find(quoted);
    if (at == std::string::npos) throw std::runtime_error("lite.json has no " + key);
    auto begin = text.find(':', at + quoted.size());
    if (begin == std::string::npos) throw std::runtime_error("lite.json: malformed " + key);
    begin = text.find_first_not_of(" \t\r\n", begin + 1);
    const auto end = text.find_first_of(",}\r\n", begin);
    std::string value = text.substr(begin, end - begin);
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.pop_back();
    if (value.size() >= 2 && value.front() == '"') value = value.substr(1, value.size() - 2);
    return value;
}

double json_number(const std::string& text, const std::string& key) { return std::stod(json_value(text, key)); }

bool json_bool(const std::string& text, const std::string& key) {
    const std::string value = json_value(text, key);
    if (value != "true" && value != "false") throw std::runtime_error("lite.json: " + key + " is not a boolean");
    return value == "true";
}

}  // namespace

Config load_config(const std::string& export_directory) {
    std::ifstream file(export_directory + "/lite.json");
    if (!file) throw std::runtime_error("Cannot open " + export_directory + "/lite.json");
    std::stringstream buffer;
    buffer << file.rdbuf();
    const std::string text = buffer.str();
    if (json_value(text, "architecture") != "lite")
        throw std::runtime_error(export_directory + " is not a HandLite export");
    const int format = static_cast<int>(json_number(text, "export_format"));
    if (format < 2 || format > kExportFormat)
        throw std::runtime_error(export_directory + " was exported for another runtime version; re-run export_lite.py");
    Config config;
    config.dim = static_cast<int>(json_number(text, "dim"));
    config.slots_per_camera = static_cast<int>(json_number(text, "slots_per_camera"));
    config.event_span_s = json_number(text, "event_span_s");
    config.calibration_head = json_bool(text, "calibration_head");
    // Written since export format 3; older exports have one token per hand.
    config.finger_tokens = text.find("\"finger_tokens\"") != std::string::npos && json_bool(text, "finger_tokens");
    // Written since export format 4.
    config.anchor_ray_depth =
        text.find("\"anchor_ray_depth\"") != std::string::npos && json_bool(text, "anchor_ray_depth");
    if (config.anchor_ray_depth) config.anchor_depth_memory_s = json_number(text, "anchor_depth_memory_s");
    config.ray_outlier_ratio =
        text.find("\"ray_outlier_ratio\"") != std::string::npos ? json_number(text, "ray_outlier_ratio") : 0.;
    config.sample_max_age_s = json_number(text, "sample_max_age_s");
    config.anchor_lookback_s = json_number(text, "anchor_lookback_s");
    config.anchor_hold_s = json_number(text, "anchor_hold_s");
    return config;
}

struct Runtime::Event {
    int camera = 0;
    double capture = 0., arrival = 0.;
    std::array<double, kHandJoints * kModelChannels> raw{};  // invalid joints zeroed
    std::array<std::uint8_t, kHandJoints> valid{};
    detail::Rays rays;  // each camera's latest ray at this event's arrival
    bool evident = false;
    detail::Sample nominal;  // triangulation of rays with the nominal calibration
    std::vector<float> tokens;  // [event_tokens][dim]
};

struct Runtime::State {
    Config config;
    std::shared_ptr<Graphs> graphs;
    double keep_s = 0.;
    std::deque<std::shared_ptr<const Event>> events;
    std::array<std::shared_ptr<const Event>, kCameras> latest{};  // newest accepted event per camera
    bool any_arrival = false;
    double last_arrival = 0.;  // kept even when every event is pruned
};

Runtime::Runtime(Config config, std::shared_ptr<Graphs> graphs) : state_(std::make_unique<State>()) {
    if (!graphs) throw std::invalid_argument("graphs is null");
    if (config.dim < 1 || config.slots_per_camera < 1) throw std::invalid_argument("Invalid config");
    state_->config = config;
    state_->graphs = std::move(graphs);
    state_->keep_s = config.context_s() + detail::kStreamKeepMargin;
}

Runtime::~Runtime() = default;
Runtime::Runtime(Runtime&&) noexcept = default;
Runtime& Runtime::operator=(Runtime&&) noexcept = default;

void Runtime::reset() {
    state_->events.clear();
    state_->latest = {};
    state_->any_arrival = false;
}

std::size_t Runtime::size() const { return state_->events.size(); }
const Config& Runtime::config() const { return state_->config; }

namespace {

std::vector<float> run(Graphs& graphs, const std::string& name, const std::vector<Tensor>& inputs,
                       std::size_t expected) {
    std::vector<float> out = graphs.run(name, inputs);
    if (out.size() != expected)
        throw std::runtime_error(name + " returned " + std::to_string(out.size()) + " values, expected " +
                                 std::to_string(expected));
    return out;
}

}  // namespace

bool Runtime::push(int camera, const float* features, const std::uint8_t* valid, double capture_time,
                   double arrival_time) {
    State& s = *state_;
    if (camera < 0 || camera >= kCameras)
        throw std::invalid_argument("camera must be in [0, 3), got " + std::to_string(camera));
    if (s.any_arrival && arrival_time < s.last_arrival) throw std::logic_error("Events must be pushed in arrival order");
    s.any_arrival = true;
    s.last_arrival = arrival_time;
    if (s.latest[camera] && capture_time <= s.latest[camera]->capture) return false;

    auto event = std::make_shared<Event>();
    event->camera = camera;
    event->capture = capture_time;
    event->arrival = arrival_time;
    for (int j = 0; j < kHandJoints; ++j) {
        event->valid[j] = valid[j] != 0;
        for (int k = 0; k < kModelChannels; ++k)
            event->raw[j * kModelChannels + k] = event->valid[j] ? features[j * kFeatureChannels + k] : 0.;
    }
    // Latest ray of each camera (this event for its own camera) captured within max age.
    for (int c = 0; c < kCameras; ++c) {
        const Event* source = c == camera ? event.get() : s.latest[c].get();
        if (!source || !(source->capture >= arrival_time - s.config.sample_max_age_s)) continue;
        event->rays.capture[c] = source->capture;
        for (int j = 0; j < kHandJoints; ++j) {
            const double* r = source->raw.data() + j * kModelChannels;
            const int index = c * kHandJoints + j;
            event->rays.origin[index] = {r[detail::kOrigin], r[detail::kOrigin + 1], r[detail::kOrigin + 2]};
            event->rays.direction[index] = {r[detail::kDirection], r[detail::kDirection + 1],
                                            r[detail::kDirection + 2]};
            event->rays.mask[index] = source->valid[j];
        }
    }
    event->nominal = detail::sample(event->rays, nullptr);
    const detail::Sample& nominal = event->nominal;
    Tensor features_in{{kHands, kJoints * kJointFeatures}, std::vector<float>(kHandJoints * kJointFeatures)};
    detail::joint_features(event->raw.data(), event->valid.data(), nominal, features_in.values.data());
    Tensor camera_in{{kCameras}, std::vector<float>(kCameras, 0.f)};
    camera_in.values[camera] = 1.f;
    event->tokens = run(*s.graphs, "encoder", {features_in, camera_in}, std::size_t(s.config.event_tokens()) * s.config.dim);
    for (int j = 0; j < kHandJoints; ++j) event->evident = event->evident || (nominal.ok[j] && event->valid[j]);

    s.latest[camera] = event;
    s.events.push_back(event);
    while (!s.events.empty() && s.events.front()->capture < arrival_time - s.keep_s) s.events.pop_front();
    return true;
}

std::array<double, kHandJoints * 3> Runtime::query(double time) { return query_details(time).pose; }

QueryResult Runtime::query_details(double time) {
    State& s = *state_;
    const Config& c = s.config;
    if (!s.any_arrival) throw std::logic_error("No events yet");
    if (time < s.last_arrival) throw std::logic_error("Query before the latest arrival");

    // Slots: each camera's K latest events arrived by time and captured within the span,
    // in arrival order, padding (nullptr) first.
    std::array<int, kCameras> taken{};
    std::vector<const Event*> chosen;
    for (auto it = s.events.rbegin(); it != s.events.rend(); ++it) {
        const Event& e = **it;
        if (e.arrival <= time + kArrivalTolerance && e.capture >= time - c.event_span_s &&
            taken[e.camera] < c.slots_per_camera) {
            ++taken[e.camera];
            chosen.push_back(&e);
        }
    }
    std::reverse(chosen.begin(), chosen.end());
    const int slots = c.slots(), per = c.event_tokens(), tokens_count = per * slots;
    std::vector<const Event*> slot(slots - chosen.size(), nullptr);
    slot.insert(slot.end(), chosen.begin(), chosen.end());

    Tensor tokens{{tokens_count, c.dim}, std::vector<float>(std::size_t(tokens_count) * c.dim, 0.f)};
    Tensor validity{{tokens_count}, std::vector<float>(tokens_count, 0.f)};
    Tensor gaps{{tokens_count}, std::vector<float>(tokens_count, 0.f)};
    for (int i = 0; i < slots; ++i) {
        if (!slot[i]) continue;
        std::copy(slot[i]->tokens.begin(), slot[i]->tokens.end(), tokens.values.begin() + std::size_t(i) * per * c.dim);
        const double gap = std::max(time - slot[i]->capture, 0.) / kTimeUnit;
        for (int t = 0; t < per; ++t) {
            validity.values[i * per + t] = 1.f;
            gaps.values[i * per + t] = static_cast<float>(gap);
        }
    }

    QueryResult result;
    if (c.calibration_head) {
        Tensor pool{{kCameras, tokens_count}, std::vector<float>(std::size_t(kCameras) * tokens_count, float(kMaskOff))};
        std::array<bool, kCameras> informative{};
        for (int i = 0; i < slots; ++i) {
            if (!slot[i]) continue;
            for (int t = 0; t < per; ++t) pool.values[slot[i]->camera * tokens_count + i * per + t] = 0.f;
            informative[slot[i]->camera] = informative[slot[i]->camera] || slot[i]->evident;
        }
        const std::vector<float> correction =
            run(*s.graphs, "calibrator", {tokens, pool}, std::size_t(kCameras) * kCalibrationParams);
        // A camera without triangulated evidence in its slots keeps the nominal calibration.
        for (int cam = 0; cam < kCameras; ++cam)
            for (int k = 0; k < kCalibrationParams; ++k)
                result.calibration[cam * kCalibrationParams + k] =
                    informative[cam] ? correction[cam * kCalibrationParams + k] : 0.;
    }

    std::vector<detail::Sample> samples;
    samples.reserve(chosen.size());
    for (const Event* e : chosen)
        samples.push_back(detail::sample(e->rays, result.calibration.data(), c.ray_outlier_ratio));
    std::vector<const detail::Sample*> ordered;
    for (const auto& sample : samples) ordered.push_back(&sample);
    // Joints seen by one camera: the newest ray of the latest slot event, and the newest
    // nominal triangulation of every stored event within the depth memory.
    detail::SingleRays single;
    const bool use_single = c.anchor_ray_depth && !chosen.empty();
    if (use_single) {
        detail::corrected_rays(chosen.back()->rays, result.calibration.data(), single);
        for (const auto& e : s.events) {
            if (!(e->arrival <= time + kArrivalTolerance)) continue;
            for (int j = 0; j < kHandJoints; ++j)
                if (e->nominal.ok[j] && e->nominal.stamp[j] >= time - c.anchor_depth_memory_s) {
                    single.reference[j] = e->nominal.point[j];
                    single.known[j] = 1;
                }
        }
    }
    const detail::Anchor anchor =
        detail::anchor_at(ordered, time, c.anchor_lookback_s, c.anchor_hold_s, use_single ? &single : nullptr);

    const int width = c.anchor_features();
    Tensor features{{kHandJoints, width}, std::vector<float>(std::size_t(kHandJoints) * width)};
    for (int j = 0; j < kHandJoints; ++j) {
        for (int k = 0; k < 3; ++k) features.values[j * width + k] = static_cast<float>(anchor.position[j][k]);
        for (int k = 0; k < width - 3; ++k) features.values[j * width + 3 + k] = static_cast<float>(anchor.flags[j][k]);
    }
    const std::vector<float> offset =
        run(*s.graphs, "decoder", {tokens, validity, gaps, features}, std::size_t(kHandJoints) * 3);
    for (int j = 0; j < kHandJoints; ++j)
        for (int k = 0; k < 3; ++k) result.pose[j * 3 + k] = anchor.position[j][k] + offset[j * 3 + k];
    return result;
}

}  // namespace hand_lite
