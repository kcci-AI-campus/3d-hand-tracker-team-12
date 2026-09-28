#include "ncnn_graphs.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

#include <net.h>

namespace hand_master {

struct NcnnGraphs::Impl {
    ncnn::Net encoder, query;
};

namespace {

void load(ncnn::Net& net, const std::string& prefix, int threads, bool fp16) {
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

// A copy in ncnn-owned memory (w = last axis): in-place layers never write into our buffers, and
// the Mat cannot outlive the data it points at.
ncnn::Mat mat(const float* data, int h, int w) {
    ncnn::Mat m;
    if (h == 1) m.create(w); else m.create(w, h);
    if (m.empty()) throw std::bad_alloc();
    std::copy(data, data + static_cast<std::size_t>(w) * h, static_cast<float*>(m.data));
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

std::vector<float> run(ncnn::Net& net, const char* name, std::vector<ncnn::Mat> inputs) {
    ncnn::Extractor extractor = net.create_extractor();
    for (std::size_t i = 0; i < inputs.size(); ++i)
        if (extractor.input(("in" + std::to_string(i)).c_str(), inputs[i]))
            throw std::runtime_error(std::string("ncnn ") + name + ": no input in" + std::to_string(i));
    ncnn::Mat output;
    if (extractor.extract("out0", output)) throw std::runtime_error(std::string("ncnn ") + name + " failed");
    return flat(output);
}

}  // namespace

NcnnGraphs::NcnnGraphs(const std::string& directory, const Config& config, int threads, bool fp16)
    : impl_(std::make_unique<Impl>()), config_(config) {
    load(impl_->encoder, directory + "/encoder", threads, fp16);
    load(impl_->query, directory + "/query", threads, fp16);
}

NcnnGraphs::~NcnnGraphs() = default;

std::vector<float> NcnnGraphs::encoder(const std::vector<float>& features, const std::array<float, 3>& camera) {
    if (features.size() != static_cast<std::size_t>(kHandJoints * kEncoderChannels))
        throw std::invalid_argument("encoder features must be [2][210]");
    return run(impl_->encoder, "encoder", {mat(features.data(), kHands, kJoints * kEncoderChannels),
                                           mat(camera.data(), 1, 3)});
}

std::vector<float> NcnnGraphs::query(const std::vector<float>& tokens, const std::vector<float>& valid,
                                     const std::vector<float>& ages) {
    const int count = config_.slots() * config_.event_tokens;
    if (tokens.size() != static_cast<std::size_t>(count) * config_.dim || valid.size() != static_cast<std::size_t>(count) ||
        ages.size() != valid.size())
        throw std::invalid_argument("query inputs must be [T][dim], [T], [T]");
    return run(impl_->query, "query", {mat(tokens.data(), count, config_.dim), mat(valid.data(), 1, count),
                                       mat(ages.data(), 1, count)});
}

}  // namespace hand_master
