#include "ncnn_graphs.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

#include <net.h>

namespace hand_master {

struct NcnnGraph::Impl {
    ncnn::Net net;
};

namespace {

// A copy in ncnn-owned memory (w = last axis): in-place layers never write into our buffers, and
// the Mat cannot outlive the data it points at.
ncnn::Mat mat(const GraphInput& input) {
    ncnn::Mat m;
    if (input.h == 1) m.create(input.w); else m.create(input.w, input.h);
    if (m.empty()) throw std::bad_alloc();
    std::copy(input.data, input.data + static_cast<std::size_t>(input.w) * input.h, static_cast<float*>(m.data));
    return m;
}

std::vector<float> flat(const ncnn::Mat& m) {
    std::vector<float> values;
    values.reserve(static_cast<std::size_t>(m.w) * m.h * m.d * m.c);
    for (int q = 0; q < m.c; ++q) {
        const ncnn::Mat channel = m.channel(q);
        for (int z = 0; z < m.d; ++z)
            for (int y = 0; y < m.h; ++y) {
                const float* row = channel.depth(z).row(y);
                values.insert(values.end(), row, row + m.w);
            }
    }
    return values;
}

}  // namespace

NcnnGraph::NcnnGraph(const std::string& prefix, int threads, bool fp16) : impl_(std::make_unique<Impl>()), name_(prefix) {
    auto& net = impl_->net;
    net.opt.use_vulkan_compute = false;
    net.opt.num_threads = threads;
    net.opt.use_fp16_packed = fp16;
    net.opt.use_fp16_storage = fp16;
    net.opt.use_fp16_arithmetic = fp16;
    for (const char* suffix : {".ncnn.param", ".ncnn.bin"}) {
        const std::string path = prefix + suffix;
        FILE* file = std::fopen(path.c_str(), "rb");
        if (!file) throw std::runtime_error("Model missing: " + path);
        const int status = std::string(suffix) == ".ncnn.param" ? net.load_param(file) : net.load_model(file);
        std::fclose(file);
        if (status) throw std::runtime_error("Cannot load ncnn graph " + path);
    }
}

NcnnGraph::~NcnnGraph() = default;

std::vector<float> NcnnGraph::run(const std::vector<GraphInput>& inputs) {
    ncnn::Extractor extractor = impl_->net.create_extractor();
    std::vector<ncnn::Mat> mats;  // alive until extract()
    mats.reserve(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        mats.push_back(mat(inputs[i]));
        if (extractor.input(("in" + std::to_string(i)).c_str(), mats.back()))
            throw std::runtime_error("ncnn " + name_ + ": no input in" + std::to_string(i));
    }
    ncnn::Mat output;
    if (extractor.extract("out0", output)) throw std::runtime_error("ncnn " + name_ + " failed");
    return flat(output);
}

NcnnGraphs::NcnnGraphs(const std::string& directory, const Config& config, int threads, bool fp16)
    : encoder_(directory + "/encoder", threads, fp16), query_(directory + "/query", threads, fp16), config_(config) {}

std::vector<float> NcnnGraphs::encoder(const std::vector<float>& features, const std::array<float, 3>& camera) {
    if (features.size() != static_cast<std::size_t>(kHandJoints * kEncoderChannels))
        throw std::invalid_argument("encoder features must be [2][210]");
    return encoder_.run({{features.data(), kHands, kJoints * kEncoderChannels}, {camera.data(), 1, 3}});
}

std::vector<float> NcnnGraphs::query(const std::vector<float>& tokens, const std::vector<float>& valid,
                                     const std::vector<float>& ages) {
    const int count = config_.slots() * config_.event_tokens;
    if (tokens.size() != static_cast<std::size_t>(count) * config_.dim || valid.size() != static_cast<std::size_t>(count) ||
        ages.size() != valid.size())
        throw std::invalid_argument("query inputs must be [T][dim], [T], [T]");
    return query_.run({{tokens.data(), count, config_.dim}, {valid.data(), 1, count}, {ages.data(), 1, count}});
}

LiteNcnnGraphs::LiteNcnnGraphs(const std::string& directory, const LiteConfig& config, int threads, bool fp16)
    : encoder_(directory + "/encoder", threads, fp16), corrector_(directory + "/corrector", threads, fp16), config_(config) {}

std::vector<float> LiteNcnnGraphs::encoder(const std::vector<float>& features, const std::array<float, 3>& camera) {
    if (features.size() != static_cast<std::size_t>(kHandJoints * kEncoderChannels))
        throw std::invalid_argument("encoder features must be [2][210]");
    return encoder_.run({{features.data(), kHands, kJoints * kEncoderChannels}, {camera.data(), 1, 3}});
}

std::vector<float> LiteNcnnGraphs::corrector(const std::vector<float>& joints, const std::vector<float>& tokens,
                                             const std::vector<float>& valid, const std::vector<float>& ages) {
    const int count = config_.slots() * config_.event_tokens;
    if (joints.size() != static_cast<std::size_t>(kHandJoints * kLiteJointFeatures) ||
        tokens.size() != static_cast<std::size_t>(count) * config_.dim || valid.size() != static_cast<std::size_t>(count) ||
        ages.size() != valid.size())
        throw std::invalid_argument("corrector inputs must be [42][56], [T][dim], [T], [T]");
    return corrector_.run({{joints.data(), kHandJoints, kLiteJointFeatures}, {tokens.data(), count, config_.dim},
                           {valid.data(), 1, count}, {ages.data(), 1, count}});
}

}  // namespace hand_master
