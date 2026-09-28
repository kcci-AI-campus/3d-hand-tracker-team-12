// The C++ HandDirect runtime replays tests/data/runtime_golden.txt (tools/make_golden.py: a real
// validation clip through the Python reference on the same ncnn graphs) and must give the same
// push results, per-hand camera counts and poses.
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "direct_runtime.hpp"
#include "ncnn_graphs.hpp"

using namespace hand_master;

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::invalid_argument("usage: runtime_test MODEL_DIR GOLDEN");
        std::ifstream file(argv[2]);
        std::string line, word;
        if (!std::getline(file, line) || line.rfind("runtime_golden 1", 0) != 0) throw std::runtime_error("Not a runtime golden file");
        const double tolerance = std::stod(line.substr(line.find("tolerance=") + 10));
        const Config config = load_config(argv[1]);
        Runtime runtime(config, std::make_shared<NcnnGraphs>(argv[1], config, 1, false));
        int pushes = 0, queries = 0;
        double worst = 0.;
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
                in >> time >> seen0 >> seen1;
                Pose expected;
                for (auto& value : expected) in >> value;
                if (!in) throw std::runtime_error("Malformed query record");
                const Pose pose = runtime.query(time);
                for (std::size_t i = 0; i < pose.size(); ++i) worst = std::max(worst, std::abs(double(pose[i]) - expected[i]));
                if (runtime.seen_by_cameras()[0] != seen0 || runtime.seen_by_cameras()[1] != seen1)
                    throw std::runtime_error("seen_by_cameras differs at query " + std::to_string(queries));
                ++queries;
            }
        }
        bool order_checked = false;
        try { runtime.query(0.); } catch (const std::logic_error&) { order_checked = true; }
        if (!order_checked) throw std::runtime_error("A query before the latest arrival must fail");
        std::cout << "runtime golden: " << pushes << " pushes, " << queries << " queries, max |pose diff| " << worst
                  << " (tolerance " << tolerance << ")\n";
        return worst <= tolerance ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
