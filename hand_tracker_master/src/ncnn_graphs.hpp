// The exported HandDirect networks (encoder.ncnn.*, query.ncnn.*) run by ncnn.
#pragma once
#include <memory>
#include <string>

#include "direct_runtime.hpp"

namespace hand_master {

class NcnnGraphs : public Graphs {
public:
    // fp16: ncnn half-precision storage/arithmetic where the CPU has it (ARMv8.2 on Pi 5); the
    // export check allows 2e-3 world units (0.8 mm) for it. threads: the graphs are small; 1-2.
    NcnnGraphs(const std::string& directory, const Config& config, int threads = 1, bool fp16 = true);
    ~NcnnGraphs() override;
    std::vector<float> encoder(const std::vector<float>& features, const std::array<float, 3>& camera) override;
    std::vector<float> query(const std::vector<float>& tokens, const std::vector<float>& valid,
                             const std::vector<float>& ages) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Config config_;
};

}  // namespace hand_master
