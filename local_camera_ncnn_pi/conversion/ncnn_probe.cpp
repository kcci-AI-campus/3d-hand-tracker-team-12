// Raw tensor probe for numerical validation with the exact deployment ncnn.
#include <net.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--version") {
      std::cout << "ncnn " << NCNN_RELEASE_TAG << "\n"; return 0;
    }
    if (argc != 6) throw std::runtime_error("probe MODEL_BASE INPUT_CHW_F32 SIZE OUTPUT_PREFIX palm|hand");
    const int size = std::stoi(argv[3]);
    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.num_threads = 2;
    if (net.load_param((std::string(argv[1])+".param").c_str()) ||
        net.load_model((std::string(argv[1])+".bin").c_str())) throw std::runtime_error("load failed");
    ncnn::Mat input(size, size, 3);
    std::ifstream file(argv[2], std::ios::binary);
    for (int c = 0; c < 3; ++c) {
      file.read(reinterpret_cast<char*>(input.channel(c).data), size * size * sizeof(float));
      if (!file) throw std::runtime_error("input read failed");
    }
    auto ex = net.create_extractor();
    if (ex.input("input", input)) throw std::runtime_error("input failed");
    const std::vector<std::string> names = std::string(argv[5]) == "palm"
      ? std::vector<std::string>{"regressors", "scores"}
      : std::vector<std::string>{"landmarks", "presence", "handedness", "world_landmarks"};
    for (const auto& name : names) {
      ncnn::Mat result;
      if (ex.extract(name.c_str(), result)) throw std::runtime_error("extract failed: " + name);
      const int count = result.w * result.h * result.d * result.c;
      auto flat = result.reshape(count);
      if (flat.empty() || flat.elempack != 1 || flat.elemsize != sizeof(float)) throw std::runtime_error("non-float output");
      std::ofstream output(std::string(argv[4])+"_"+name+".f32", std::ios::binary);
      output.write(reinterpret_cast<const char*>(flat.data), count * sizeof(float));
      if (!output) throw std::runtime_error("output write failed");
      std::cout << name << " " << count << "\n";
    }
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
