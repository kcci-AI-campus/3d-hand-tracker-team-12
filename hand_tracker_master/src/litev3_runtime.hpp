// HandLiteV3 runtime (C++17, standard library only): the port of the Python LiteV3Runtime
// (hand_tracking/runtime.py). Per camera and joint a least-squares line over time through the ray
// directions of the recent detections, read at the query and triangulated; a joint that fails is
// anchored from its own last triangulation (history search, then this runtime's state). A corrector
// network predicts each joint's offset from its anchor, its expected error and each hand's in-view
// logit. The geometry runs in double precision like the Python (numpy float64) reference.
#pragma once
#include <array>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "direct_runtime.hpp"

namespace hand_master {

constexpr int kLiteJointFeatures = 56;   // corrector input per joint: 11 of the joint + 15 per camera
constexpr int kLiteRawChannels = 11;     // the event features the geometry reads (model channels)
constexpr double kMissScale = 10., kResidualUnit = .01, kMinTimeStdS = .005, kMinRaySin2 = 2e-3, kFar = 1e6;

// How each joint's anchor was made (model.ANCHOR_KINDS).
enum AnchorKind : std::uint8_t { kAnchorNone = 0, kAnchorTriangulated = 1, kAnchorRay = 2, kAnchorPrior = 3 };

struct LiteConfig {
    int dim = 64, slots_per_camera = 8, event_tokens = 10;
    double fit_span_s = .2, prior_span_s = 1., prior_step_s = .2, event_span_s = .5;
    bool error_estimate = true, presence = true;
    int prior_steps() const;
    double max_prior_age_s() const { return prior_steps() * prior_step_s + fit_span_s; }
    double context_s() const;
    int slots() const { return kCameras * slots_per_camera; }
    int output_columns() const { return 3 + (error_estimate ? 1 : 0) + (presence ? 1 : 0); }
};

// litev3.json of a HandLiteV3 export (training.export, export format 4).
LiteConfig load_lite_config(const std::string& export_directory);

class LiteGraphs {
public:
    virtual ~LiteGraphs() = default;
    // features [2][210], camera one-hot [3] -> tokens [event_tokens][dim]
    virtual std::vector<float> encoder(const std::vector<float>& features, const std::array<float, 3>& camera) = 0;
    // joint features [42][56], tokens [T][dim], validity [T], ages [T] -> [42][output_columns]
    virtual std::vector<float> corrector(const std::vector<float>& joints, const std::vector<float>& tokens,
                                         const std::vector<float>& valid, const std::vector<float>& ages) = 0;
};

class LiteV3Runtime : public PoseRuntime {
public:
    LiteV3Runtime(LiteConfig config, std::shared_ptr<LiteGraphs> graphs);

    bool push(int camera, const Features& features, const Valid& valid, double capture, double arrival) override;
    Pose query(double time) override;
    double last_arrival() const override { return last_arrival_; }
    std::size_t size() const override { return events_.size(); }
    const std::array<int, kHands>& seen_by_cameras() const override { return seen_; }
    std::optional<std::array<float, kHands>> in_view_probability() const override;

    // After the last query: each joint's expected error in mm (error_estimate) and anchor kind.
    const std::array<float, kHandJoints>& error_mm() const { return error_mm_; }
    const std::array<std::uint8_t, kHandJoints>& anchor_kind() const { return anchor_kind_; }
    const LiteConfig& config() const { return config_; }

    struct Event {
        int camera;
        double capture, arrival;
        std::array<double, kHandJoints * kLiteRawChannels> raw;  // [42][11], undetected joints zeroed
        std::array<std::uint8_t, kHandJoints> valid;
        std::vector<float> tokens;                                // [event_tokens][dim]
        std::array<std::uint8_t, kHands> detected;
    };

private:
    LiteConfig config_;
    std::shared_ptr<LiteGraphs> graphs_;
    std::deque<Event> events_;
    std::array<double, kCameras> newest_capture_{};
    std::array<bool, kCameras> has_capture_{};
    double last_arrival_ = 0.;
    bool has_arrival_ = false;
    // State: each joint's last triangulation and the capture time of the newest detection it used.
    std::array<std::array<double, 3>, kHandJoints> prior_point_{};
    std::array<double, kHandJoints> prior_capture_{};
    std::array<bool, kHandJoints> prior_known_{};
    std::array<int, kHands> seen_{};
    std::array<float, kHandJoints> error_mm_{};
    std::array<std::uint8_t, kHandJoints> anchor_kind_{};
    std::array<float, kHands> in_view_{};
    bool has_in_view_ = false;
};

}  // namespace hand_master
