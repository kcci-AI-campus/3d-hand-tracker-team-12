// The C++ HandLiteV3 runtime replays tests/data/litev3_golden.txt (tools/make_golden.py: a real
// validation clip and a camera-0-only stretch through the repository's Python LiteV3Runtime on the
// same ncnn graphs) and must give the same push results, anchor kinds, per-hand camera counts,
// in-view probabilities, expected errors and poses.
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "litev3_runtime.hpp"
#include "ncnn_graphs.hpp"

using namespace hand_master;

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::invalid_argument("usage: litev3_test MODEL_DIR GOLDEN");
        std::ifstream file(argv[2]);
        std::string line, word;
        if (!std::getline(file, line) || line.rfind("litev3_golden 1", 0) != 0) throw std::runtime_error("Not a HandLiteV3 golden file");
        const double tolerance = std::stod(line.substr(line.find("tolerance=") + 10));
        const LiteConfig config = load_lite_config(argv[1]);
        LiteV3Runtime runtime(config, std::make_shared<LiteNcnnGraphs>(argv[1], config, 1, false));
        int pushes = 0, queries = 0, kinds = 0;
        double pose_worst = 0., view_worst = 0., error_worst = 0.;
        while (std::getline(file, line)) {
            std::istringstream in(line);
            in >> word;
            if (word == "push") {
                int camera, expected;
                double capture, arrival;
                Features features;
                Valid valid;
                in >> camera >> capture >> arrival >> expected;
                for (auto& value : features) in >> value;
                for (auto& value : valid) { int v; in >> v; value = static_cast<std::uint8_t>(v); }
                if (!in) throw std::runtime_error("Malformed push record");
                if (runtime.push(camera, features, valid, capture, arrival) != (expected != 0))
                    throw std::runtime_error("push result differs at record " + std::to_string(pushes));
                ++pushes;
            } else if (word == "query") {
                double time;
                int seen0, seen1;
                std::array<int, kHandJoints> kind;
                std::array<double, kHands> in_view;
                std::array<double, kHandJoints> error_log;
                std::array<double, kHandJoints * 3> expected;
                in >> time >> seen0 >> seen1;
                for (auto& value : kind) in >> value;
                for (auto& value : in_view) in >> value;
                for (auto& value : error_log) in >> value;
                for (auto& value : expected) in >> value;
                if (!in) throw std::runtime_error("Malformed query record");
                const Pose pose = runtime.query(time);
                for (int j = 0; j < kHandJoints; ++j) {
                    if (runtime.anchor_kind()[j] != kind[j])
                        throw std::runtime_error("anchor kind differs at query " + std::to_string(queries) + ", joint " + std::to_string(j));
                    error_worst = std::max(error_worst, std::abs(std::log1p(double(runtime.error_mm()[j])) - error_log[j]));
                }
                kinds += kHandJoints;
                for (std::size_t i = 0; i < pose.size(); ++i) pose_worst = std::max(pose_worst, std::abs(double(pose[i]) - expected[i]));
                const auto view = runtime.in_view_probability();
                if (!view) throw std::runtime_error("no in-view probability");
                for (int h = 0; h < kHands; ++h) view_worst = std::max(view_worst, std::abs(double((*view)[h]) - in_view[h]));
                if (runtime.seen_by_cameras()[0] != seen0 || runtime.seen_by_cameras()[1] != seen1)
                    throw std::runtime_error("seen_by_cameras differs at query " + std::to_string(queries));
                ++queries;
            }
        }
        std::cout << "HandLiteV3 golden: " << pushes << " pushes, " << queries << " queries, " << kinds
                  << " anchor kinds equal; max |pose diff| " << pose_worst << ", |in-view diff| " << view_worst
                  << ", |log(1+error mm) diff| " << error_worst << " (tolerance " << tolerance << ")\n";
        return pose_worst <= tolerance && view_worst <= tolerance && error_worst <= tolerance ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
