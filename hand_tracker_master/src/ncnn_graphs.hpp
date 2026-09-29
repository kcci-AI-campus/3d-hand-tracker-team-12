// Exported networks run by ncnn: one graph (NcnnGraph), HandDirect's encoder + query
// (NcnnGraphs) and HandLiteV3's encoder + corrector (LiteNcnnGraphs).
#pragma once
#include <memory>
#include <string>
#include <vector>

#include "direct_runtime.hpp"
#include "litev3_runtime.hpp"

namespace hand_master {

// A 1- or 2-axis float input: h rows of w values (h == 1: a vector of w).
struct GraphInput {
    const float* data;
    int h, w;
};

class NcnnGraph {
public:
    // prefix: path without ".ncnn.param"/".ncnn.bin". fp16: half-precision storage/arithmetic where
    // the CPU has it (ARMv8.2 on Pi 5). threads: the graphs are small; 1-2.
    NcnnGraph(const std::string& prefix, int threads, bool fp16);
    ~NcnnGraph();
    NcnnGraph(const NcnnGraph&) = delete;
    NcnnGraph& operator=(const NcnnGraph&) = delete;
    // Inputs in0.., output out0 flattened row-major.
    std::vector<float> run(const std::vector<GraphInput>& inputs);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string name_;
};

class NcnnGraphs : public Graphs {
public:
    NcnnGraphs(const std::string& directory, const Config& config, int threads = 1, bool fp16 = true);
    std::vector<float> encoder(const std::vector<float>& features, const std::array<float, 3>& camera) override;
    std::vector<float> query(const std::vector<float>& tokens, const std::vector<float>& valid,
                             const std::vector<float>& ages) override;

private:
    NcnnGraph encoder_, query_;
    Config config_;
};

class LiteNcnnGraphs : public LiteGraphs {
public:
    LiteNcnnGraphs(const std::string& directory, const LiteConfig& config, int threads = 1, bool fp16 = true);
    std::vector<float> encoder(const std::vector<float>& features, const std::array<float, 3>& camera) override;
    std::vector<float> corrector(const std::vector<float>& joints, const std::vector<float>& tokens,
                                 const std::vector<float>& valid, const std::vector<float>& ages) override;

private:
    NcnnGraph encoder_, corrector_;
    LiteConfig config_;
};

}  // namespace hand_master
