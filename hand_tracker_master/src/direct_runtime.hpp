// HandDirect runtime (C++17, standard library only): the port of the Python DirectRuntime
// (hand_tracking/direct_runtime.py). Each arriving camera frame is encoded once into finger
// tokens; a query picks each camera's latest slots_per_camera events captured within
// event_span_s and runs the query network. The networks come through the Graphs interface
// (ncnn_graphs.hpp on the device).
#pragma once
#include <array>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace hand_master {

constexpr int kCameras = 3, kHands = 2, kJoints = 21, kHandJoints = kHands * kJoints;
constexpr int kFeatureChannels = 14;  // event features per joint (the dataset's layout)
constexpr int kEncoderChannels = 10;  // u,v, ray origin, ray direction, delay, detected
constexpr double kTimeUnitS = .1;     // ages and delays enter the networks in 0.1 s units
constexpr double kArrivalToleranceS = 1e-6, kStreamKeepMarginS = .5, kArrivalMarginS = .25;

using Features = std::array<float, kHandJoints * kFeatureChannels>;  // [2][21][14]
using Valid = std::array<std::uint8_t, kHandJoints>;                // [2][21], 0/1
using Pose = std::array<float, kHandJoints * 3>;                    // [2][21][3] world units

struct Config {
    int dim = 64, slots_per_camera = 8, event_tokens = 10;
    double event_span_s = .5;
    double context_s() const { return event_span_s + kArrivalMarginS; }
    int slots() const { return kCameras * slots_per_camera; }
};

// direct.json of a HandDirect export (training.export): architecture "direct".
Config load_config(const std::string& export_directory);

// The two exported networks. Inputs and outputs are row-major float arrays.
class Graphs {
public:
    virtual ~Graphs() = default;
    // features [2][210], camera one-hot [3] -> tokens [event_tokens][dim]
    virtual std::vector<float> encoder(const std::vector<float>& features, const std::array<float, 3>& camera) = 0;
    // tokens [T][dim], validity [T] (0/1), ages [T] (0.1 s units) -> joints [42][3]; T = event_tokens * slots
    virtual std::vector<float> query(const std::vector<float>& tokens, const std::vector<float>& valid,
                                     const std::vector<float>& ages) = 0;
};

class Runtime {
public:
    Runtime(Config config, std::shared_ptr<Graphs> graphs);

    // One arrived camera frame (absolute seconds). false (ignored) when its capture is no newer
    // than that camera's latest. Frames must be pushed in arrival order; frames without
    // detections count (they replace the camera's older frame in the slots).
    bool push(int camera, const Features& features, const Valid& valid, double capture, double arrival);

    // Pose [2][21][3] at time (not before the latest arrival), world units.
    Pose query(double time);

    // Per hand, after the last query: how many cameras had a frame in the query's slots that
    // detected at least kSeenMinJoints of that hand's joints (0: nothing shows the hand, so its
    // joints are the network's guess).
    static constexpr int kSeenMinJoints = 5;
    const std::array<int, kHands>& seen_by_cameras() const { return seen_; }

    std::size_t size() const { return events_.size(); }
    double last_arrival() const { return last_arrival_; }
    const Config& config() const { return config_; }

private:
    struct Event {
        int camera;
        double capture, arrival;
        std::vector<float> tokens;  // [event_tokens][dim]
        std::array<std::uint8_t, kHands> detected;  // detected joints per hand
    };
    Config config_;
    std::shared_ptr<Graphs> graphs_;
    std::deque<Event> events_;
    std::array<double, kCameras> newest_capture_;
    std::array<bool, kCameras> has_capture_{};
    double last_arrival_ = 0.;
    bool has_arrival_ = false;
    std::array<int, kHands> seen_{};
};

// The encoder input of one event: features [2][21][14] (invalid joints ignored) -> [2][210].
std::vector<float> encoder_input(const Features& features, const Valid& valid);

}  // namespace hand_master
