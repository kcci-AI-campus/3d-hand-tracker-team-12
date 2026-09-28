// Graphs backed by ncnn: loads <directory>/{encoder,calibrator,decoder}.ncnn.{param,bin}
// written by export_lite.py. Build with -DHAND_LITE_WITH_NCNN=ON.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "hand_lite/hand_lite.hpp"

namespace hand_lite {

class NcnnGraphs : public Graphs {
public:
    // One thread was fastest for these small graphs on x86; fp16 may be enabled by
    // default on ARM, so check accuracy on the device (tests/golden_test --ncnn).
    explicit NcnnGraphs(const std::string& directory, int threads = 1, bool fp16 = true);
    ~NcnnGraphs() override;
    std::vector<float> run(const std::string& name, const std::vector<Tensor>& inputs) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace hand_lite
