// Replays tests/data/golden.txt (tools/make_golden.py, recorded from the Python runtime).
//
//   golden_test <data dir>          ReplayGraphs returns the recorded network outputs and
//                                   checks every network input C++ builds, push results,
//                                   errors, poses and calibrations: the logic around the
//                                   graphs must match Python exactly (to float rounding).
//   golden_test <data dir> --ncnn   (HAND_LITE_WITH_NCNN) runs the real ncnn graphs and
//                                   checks poses within 2e-3 world units (ncnn fp16/GELU).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "hand_lite/hand_lite.hpp"
#ifdef HAND_LITE_WITH_NCNN
#include "hand_lite/ncnn_graphs.hpp"
#endif

namespace {

using hand_lite::Tensor;

struct Call {
    std::string name;
    std::vector<Tensor> inputs;
    Tensor output;
};

struct Operation {
    bool push = false;
    int camera = 0;
    double capture = 0., arrival = 0., time = 0.;
    std::vector<float> features;
    std::vector<std::uint8_t> valid;
    std::vector<Call> calls;
    std::string result;             // "0", "1", "ok", "error_invalid", "error_logic"
    std::vector<double> expected;   // pose (126) then calibration (18)
};

class Reader {
public:
    explicit Reader(const std::string& path) : file_(path) {
        if (!file_) throw std::runtime_error("Cannot open " + path);
    }
    std::string word() {
        std::string value;
        if (!(file_ >> value)) throw std::runtime_error("Unexpected end of golden file");
        return value;
    }
    void expect(const std::string& value) {
        const std::string actual = word();
        if (actual != value) throw std::runtime_error("Golden file: expected " + value + ", got " + actual);
    }
    double number() { return std::stod(word()); }
    int integer() { return std::stoi(word()); }
    Tensor tensor() {
        Tensor t;
        const int dims = integer();
        std::size_t size = 1;
        for (int i = 0; i < dims; ++i) {
            t.shape.push_back(integer());
            size *= t.shape.back();
        }
        t.values.resize(size);
        for (auto& v : t.values) v = static_cast<float>(number());
        return t;
    }

private:
    std::ifstream file_;
};

std::vector<Operation> read_golden(const std::string& path) {
    Reader in(path);
    in.expect("hand_lite_golden");
    in.expect("1");
    std::vector<Operation> operations;
    for (std::string token = in.word(); token != "end"; token = in.word()) {
        if (token != "op") throw std::runtime_error("Golden file: expected op, got " + token);
        Operation op;
        const std::string kind = in.word();
        op.push = kind == "push";
        if (op.push) {
            op.camera = in.integer();
            op.capture = in.number();
            op.arrival = in.number();
            op.features.resize(hand_lite::kHandJoints * hand_lite::kFeatureChannels);
            for (auto& v : op.features) v = static_cast<float>(in.number());
            op.valid.resize(hand_lite::kHandJoints);
            for (auto& v : op.valid) v = static_cast<std::uint8_t>(in.integer());
        } else {
            op.time = in.number();
        }
        in.expect("calls");
        const int calls = in.integer();
        for (int i = 0; i < calls; ++i) {
            in.expect("call");
            Call call;
            call.name = in.word();
            const int inputs = in.integer();
            for (int k = 0; k < inputs; ++k) call.inputs.push_back(in.tensor());
            call.output = in.tensor();
            op.calls.push_back(std::move(call));
        }
        in.expect("result");
        op.result = in.word();
        if (op.result == "ok")
            for (int i = 0; i < hand_lite::kHandJoints * 3 + hand_lite::kCameras * hand_lite::kCalibrationParams; ++i)
                op.expected.push_back(in.number());
        operations.push_back(std::move(op));
    }
    return operations;
}

// Returns recorded outputs in order; fails on a different call or input.
class ReplayGraphs : public hand_lite::Graphs {
public:
    std::deque<Call> pending;
    double worst_input = 0.;

    std::vector<float> run(const std::string& name, const std::vector<Tensor>& inputs) override {
        if (pending.empty()) throw std::runtime_error("Unexpected " + name + " call");
        Call call = std::move(pending.front());
        pending.pop_front();
        if (call.name != name) throw std::runtime_error("Called " + name + ", Python called " + call.name);
        if (call.inputs.size() != inputs.size()) throw std::runtime_error(name + ": input count differs");
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            if (inputs[i].shape != call.inputs[i].shape)
                throw std::runtime_error(name + ": shape of input " + std::to_string(i) + " differs");
            for (std::size_t k = 0; k < inputs[i].values.size(); ++k) {
                const double difference = std::abs(double(inputs[i].values[k]) - call.inputs[i].values[k]);
                if (!(difference <= 1e-4))
                    throw std::runtime_error(name + " input " + std::to_string(i) + "[" + std::to_string(k) +
                                             "]: C++ " + std::to_string(inputs[i].values[k]) + ", Python " +
                                             std::to_string(call.inputs[i].values[k]));
                worst_input = std::max(worst_input, difference);
            }
        }
        return call.output.values;
    }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: golden_test <data dir> [--ncnn]\n");
        return 2;
    }
    const std::string directory = argv[1];
    const bool ncnn = argc > 2 && std::string(argv[2]) == "--ncnn";
    try {
        const auto operations = read_golden(directory + "/golden.txt");
        auto replay = std::make_shared<ReplayGraphs>();
        std::shared_ptr<hand_lite::Graphs> graphs = replay;
        if (ncnn) {
#ifdef HAND_LITE_WITH_NCNN
            graphs = std::make_shared<hand_lite::NcnnGraphs>(directory);
#else
            throw std::runtime_error("Built without HAND_LITE_WITH_NCNN");
#endif
        }
        hand_lite::Runtime runtime(hand_lite::load_config(directory), graphs);
        const double tolerance = ncnn ? 2e-3 : 1e-6;
        double worst_pose = 0., worst_calibration = 0.;
        int pushes = 0, queries = 0;
        for (std::size_t n = 0; n < operations.size(); ++n) {
            const Operation& op = operations[n];
            const std::string where = "operation " + std::to_string(n) + (op.push ? " (push)" : " (query)");
            if (!ncnn) replay->pending.assign(op.calls.begin(), op.calls.end());
            std::string outcome;
            hand_lite::QueryResult result;
            try {
                if (op.push) {
                    outcome = runtime.push(op.camera, op.features.data(), op.valid.data(), op.capture, op.arrival) ? "1" : "0";
                    ++pushes;
                } else {
                    result = runtime.query_details(op.time);
                    outcome = "ok";
                    ++queries;
                }
            } catch (const std::invalid_argument&) {
                outcome = "error_invalid";
            } catch (const std::logic_error&) {
                outcome = "error_logic";
            }
            if (outcome != op.result) throw std::runtime_error(where + ": " + outcome + ", Python " + op.result);
            if (!ncnn && !replay->pending.empty()) throw std::runtime_error(where + ": fewer network calls than Python");
            if (outcome != "ok") continue;
            for (int i = 0; i < hand_lite::kHandJoints * 3; ++i) {
                const double difference = std::abs(result.pose[i] - op.expected[i]);
                if (!(difference <= tolerance))
                    throw std::runtime_error(where + ": pose[" + std::to_string(i) + "] differs by " + std::to_string(difference));
                worst_pose = std::max(worst_pose, difference);
            }
            for (int i = 0; i < hand_lite::kCameras * hand_lite::kCalibrationParams; ++i) {
                const double difference = std::abs(result.calibration[i] - op.expected[hand_lite::kHandJoints * 3 + i]);
                if (!(difference <= tolerance))
                    throw std::runtime_error(where + ": calibration[" + std::to_string(i) + "] differs by " + std::to_string(difference));
                worst_calibration = std::max(worst_calibration, difference);
            }
        }
        std::printf("golden_test %s: %d pushes, %d queries, %zu operations OK; max |pose diff| %.3g, "
                    "max |calibration diff| %.3g, max |network input diff| %.3g\n",
                    ncnn ? "ncnn" : "replay", pushes, queries, operations.size(), worst_pose, worst_calibration,
                    replay->worst_input);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "golden_test FAILED: %s\n", error.what());
        return 1;
    }
}
