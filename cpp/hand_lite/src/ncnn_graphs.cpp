#include "hand_lite/ncnn_graphs.hpp"

#include <map>
#include <stdexcept>

#include <net.h>  // ncnn

namespace hand_lite {

struct NcnnGraphs::Impl {
    std::map<std::string, std::unique_ptr<ncnn::Net>> nets;
};

NcnnGraphs::NcnnGraphs(const std::string& directory, int threads, bool fp16) : impl_(std::make_unique<Impl>()) {
    for (const char* name : {"encoder", "calibrator", "decoder"}) {
        auto net = std::make_unique<ncnn::Net>();
        net->opt.use_vulkan_compute = false;
        net->opt.num_threads = threads;
        net->opt.use_fp16_packed = fp16;
        net->opt.use_fp16_storage = fp16;
        net->opt.use_fp16_arithmetic = fp16;
        const std::string prefix = directory + "/" + name;
        if (net->load_param((prefix + ".ncnn.param").c_str()) || net->load_model((prefix + ".ncnn.bin").c_str()))
            throw std::runtime_error("Cannot load ncnn graph " + prefix);
        impl_->nets[name] = std::move(net);
    }
}

NcnnGraphs::~NcnnGraphs() = default;

namespace {

// A 1- or 2-axis Tensor (all HandLite inputs) -> an ncnn-owned Mat with w = last axis.
// Owned memory also keeps in-place layers from writing into the caller's buffer.
ncnn::Mat to_mat(const Tensor& tensor) {
    const auto& s = tensor.shape;
    ncnn::Mat mat;
    if (s.size() == 1) mat.create(s[0]);
    else if (s.size() == 2) mat.create(s[1], s[0]);
    else throw std::invalid_argument("HandLite graph inputs have 1 or 2 axes");
    if (mat.empty() || std::size_t(mat.w) * mat.h != tensor.values.size())
        throw std::invalid_argument("Tensor shape and size differ");
    std::copy(tensor.values.begin(), tensor.values.end(), static_cast<float*>(mat.data));
    return mat;
}

std::vector<float> from_mat(const ncnn::Mat& mat) {
    std::vector<float> values;
    values.reserve(std::size_t(mat.w) * mat.h * mat.d * mat.c);
    for (int q = 0; q < mat.c; ++q) {
        const ncnn::Mat channel = mat.channel(q);
        for (int z = 0; z < mat.d; ++z)
            for (int y = 0; y < mat.h; ++y) {
                const float* row = channel.depth(z).row(y);
                values.insert(values.end(), row, row + mat.w);
            }
    }
    return values;
}

}  // namespace

std::vector<float> NcnnGraphs::run(const std::string& name, const std::vector<Tensor>& inputs) {
    const auto net = impl_->nets.find(name);
    if (net == impl_->nets.end()) throw std::invalid_argument("Unknown graph " + name);
    ncnn::Extractor extractor = net->second->create_extractor();
    std::vector<ncnn::Mat> mats;  // alive until extract()
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        mats.push_back(to_mat(inputs[i]));
        if (extractor.input(("in" + std::to_string(i)).c_str(), mats.back()))
            throw std::runtime_error("ncnn " + name + ": no input in" + std::to_string(i));
    }
    ncnn::Mat output;
    if (extractor.extract("out0", output)) throw std::runtime_error("ncnn " + name + " failed");
    return from_mat(output);
}

}  // namespace hand_lite
