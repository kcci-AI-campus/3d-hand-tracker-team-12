// HandLite deployment runtime: the geometry and event logic around the three exported
// networks (encoder, calibrator, decoder). C++ port of hand_tracking/lite_runtime.py;
// tests/golden_test.cpp checks it against that reference.
//
//   auto graphs = std::make_shared<hand_lite::NcnnGraphs>("deploy");   // ncnn_graphs.hpp
//   hand_lite::Runtime runtime(hand_lite::load_config("deploy"), graphs);
//   runtime.push(camera, features, valid, capture_time, arrival_time);  // each camera frame
//   std::array<double, 126> pose = runtime.query(time);                 // [2][21][3] world units
//
// Times are absolute seconds. Frames must be pushed in arrival order.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace hand_lite {

constexpr int kCameras = 3;
constexpr int kHands = 2;
constexpr int kJoints = 21;
constexpr int kHandJoints = kHands * kJoints;
constexpr int kFingers = 5;
// Exported per-joint channels of an event; see hand_tracking/constants.py.
constexpr int kFeatureChannels = 14;
constexpr int kModelChannels = 11;
constexpr int kJointFeatures = 14;
constexpr int kCalibrationParams = 6;
// Graph interface version written by export_lite.py (lite.json "export_format"). Versions 2
// and 3 are still read, with the options they predate (finger_tokens, anchor_ray_depth) off.
constexpr int kExportFormat = 4;

struct Config {
    int dim = 64;
    int slots_per_camera = 8;
    double event_span_s = .5;
    bool calibration_head = true;
    bool finger_tokens = true;  // encoder tokens per event: kHands * kFingers, else kHands
    double sample_max_age_s = .2;
    double anchor_lookback_s = .2;
    double anchor_hold_s = .3;
    // A joint seen by fewer than two cameras is anchored on its newest ray at a remembered depth.
    bool anchor_ray_depth = true;
    double anchor_depth_memory_s = 1.;
    // Query-time triangulation drops one ray this many times less consistent than the others (0: off).
    double ray_outlier_ratio = 0.;

    int slots() const { return kCameras * slots_per_camera; }
    int event_tokens() const { return kHands * (finger_tokens ? kFingers : 1); }
    // Decoder input per joint: anchor (3), flags (5) and the single-ray flag.
    int anchor_features() const { return 8 + (anchor_ray_depth ? 1 : 0); }
    // Seconds of history a query can reach (hand_tracking.config.LiteConfig.context_s).
    double context_s() const;
};

// Reads <export_directory>/lite.json; throws std::runtime_error for another architecture
// or export format.
Config load_config(const std::string& export_directory);

// A graph input or output without a batch axis, row-major.
struct Tensor {
    std::vector<int> shape;
    std::vector<float> values;
};

// Runs the exported networks. Inputs and outputs (all float32, row-major), P = event_tokens():
//   "encoder"    features [2, 21*14], camera one-hot [3]                   -> tokens [P, dim]
//   "calibrator" tokens [P*slots, dim], pool bias [3, P*slots]            -> correction [3, 6]
//   "decoder"    tokens [P*slots, dim], token validity [P*slots] (0/1),
//                gaps [P*slots] (0.1 s units), anchor features [42, A]    -> offset [42, 3]
// A = anchor_features().
class Graphs {
public:
    virtual ~Graphs() = default;
    virtual std::vector<float> run(const std::string& name, const std::vector<Tensor>& inputs) = 0;
};

struct QueryResult {
    std::array<double, kHandJoints * 3> pose{};                       // [2][21][3]
    std::array<double, kCameras * kCalibrationParams> calibration{};  // [3][6] used for the anchor
};

class Runtime {
public:
    Runtime(Config config, std::shared_ptr<Graphs> graphs);
    ~Runtime();
    Runtime(Runtime&&) noexcept;
    Runtime& operator=(Runtime&&) noexcept;

    // features [2][21][14] and valid [2][21] (nonzero = detected) are copied. Returns false
    // (ignored) for a capture no newer than that camera's latest; frames without any
    // detection count. Throws std::invalid_argument for a camera outside [0, 3) and
    // std::logic_error for an arrival earlier than the previous one.
    bool push(int camera, const float* features, const std::uint8_t* valid, double capture_time,
              double arrival_time);
    // Pose [2][21][3] at time (not before the latest arrival); throws std::logic_error otherwise.
    std::array<double, kHandJoints * 3> query(double time);
    QueryResult query_details(double time);

    void reset();
    std::size_t size() const;  // stored events
    const Config& config() const;

private:
    struct Event;
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace hand_lite
