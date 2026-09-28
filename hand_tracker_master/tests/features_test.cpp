// make_features on models/rig.json must reproduce the dataset's event features from the slave-format
// u,v (tests/data/features_golden.txt, tools/make_golden.py): u,v, ray origin and direction, delay.
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "rig.hpp"

using namespace hand_master;

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::invalid_argument("usage: features_test RIG_JSON GOLDEN");
        const Rig rig = load_rig(argv[1]);
        std::ifstream file(argv[2]);
        std::string line, word;
        if (!std::getline(file, line) || line.rfind("features_golden 1", 0) != 0) throw std::runtime_error("Not a features golden file");
        const double tolerance = std::stod(line.substr(line.find("tolerance=") + 10));
        int events = 0;
        double worst = 0.;
        while (std::getline(file, line)) {
            std::istringstream in(line);
            int camera;
            double delay;
            std::array<float, 84> uv;
            Features expected;
            Valid expected_valid;
            in >> word >> camera >> delay;
            for (auto& value : uv) in >> value;
            for (auto& value : expected) in >> value;
            for (auto& value : expected_valid) { int v; in >> v; value = static_cast<std::uint8_t>(v); }
            if (!in || word != "event") throw std::runtime_error("Malformed features record");
            Features features;
            Valid valid;
            make_features(rig, camera, uv, delay, features, valid);
            if (valid != expected_valid) throw std::runtime_error("validity differs at event " + std::to_string(events));
            for (int j = 0; j < kHandJoints; ++j)
                for (int c : {0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13})  // 8, 9: times relative to a query, not inputs
                    worst = std::max(worst, std::abs(double(features[j * kFeatureChannels + c]) - expected[j * kFeatureChannels + c]));
            ++events;
        }
        // Undetected (-1), NaN and out-of-image joints are all invalid.
        std::array<float, 84> uv;
        uv.fill(-1.f);
        uv[0] = std::nanf("");
        uv[2] = 1.5f;
        Features features;
        Valid valid;
        make_features(rig, 0, uv, .1, features, valid);
        for (auto v : valid) if (v) throw std::runtime_error("an undetected joint became valid");
        std::cout << "rig features: " << events << " events, max |diff| " << worst << " (tolerance " << tolerance << ")\n";
        return events > 0 && worst <= tolerance ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
