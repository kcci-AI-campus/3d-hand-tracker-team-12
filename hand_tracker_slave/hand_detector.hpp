// CPU hand pipeline: palm detection -> rotated crop -> 21 landmarks (image pixels).
// Models converted directly from Google's official hand_landmarker.task; see
// MODEL_STRUCTURE.md and conversion/parity_report.json. Shared by the slave app and
// hand_tracker_master (the master Pi's own camera).
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <net.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace hand_detector {
namespace fs = std::filesystem;
constexpr int kPalm = 192, kHand = 224;
constexpr float kPi = 3.14159265358979323846f;

struct Palm {
  cv::Rect2f box;
  std::array<cv::Point2f, 7> keypoints;
  float confidence;
};
struct Hand {
  std::array<cv::Point3f, 21> points;  // image pixels, including normalized-Z correction.
  std::array<cv::Point3f, 21> world_points;  // model-relative metres, rotation corrected.
  float palm_confidence, presence, right_probability;
};
struct LandmarkResult {
  std::array<float, 63> points, world;
  float presence, right_probability;
};
void Require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}

// Raspberry Pi OS uses UTF-8 filesystem paths.
void Load(ncnn::Net& net, const fs::path& base) {
  for (const auto& suffix : {".param", ".bin"}) {
    fs::path path = base;
    path += suffix;
    FILE* file = std::fopen(path.string().c_str(), "rb");
    Require(file != nullptr, "Model missing: " + path.u8string());
    const int status = std::string(suffix) == ".param" ? net.load_param(file) : net.load_model(file);
    std::fclose(file);
    Require(status == 0, "Cannot load model: " + path.u8string());
  }
}

ncnn::Mat Input(const cv::Mat& bgr) {
  Require(bgr.type() == CV_8UC3 && bgr.isContinuous(), "Expected contiguous BGR image");
  auto input = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows);
  const float norm[3] = {1.f / 255, 1.f / 255, 1.f / 255};
  input.substract_mean_normalize(nullptr, norm);
  return input;
}

float IoU(const cv::Rect2f& a, const cv::Rect2f& b) {
  const float overlap = (a & b).area();
  return overlap / std::max(1e-6f, a.area() + b.area() - overlap);
}

class HandDetector {
 public:
  HandDetector(const fs::path& models, int max_hands, int threads) : max_hands_(max_hands) {
    for (auto* net : {&palm_, &landmark_}) {
      net->opt.num_threads = threads;
      net->opt.use_vulkan_compute = false;
      net->opt.use_fp16_packed = false;
      net->opt.use_fp16_storage = false;
      net->opt.use_fp16_arithmetic = false;
    }
    // The new converter emits InnerProduct heads; channel packing is supported.
    Load(palm_, models / "hand_detector");
    Load(landmark_, models / "hand_landmarks_detector");
    // Fixed-size SSD anchors: stride 8 (2 per cell), merged stride 16 (6 per cell).
    for (int stride : {8, 16}) {
      const int cells = kPalm / stride, repeats = stride == 8 ? 2 : 6;
      for (int y = 0; y < cells; ++y)
        for (int x = 0; x < cells; ++x)
          for (int a = 0; a < repeats; ++a)
            anchors_.emplace_back((x + 0.5f) * stride, (y + 0.5f) * stride);
    }
  }

  std::vector<Hand> Detect(const cv::Mat& bgr) {
    const float scale = static_cast<float>(kPalm) / std::max(bgr.cols, bgr.rows);
    const int width = std::max(1, static_cast<int>(std::round(bgr.cols * scale)));
    const int height = std::max(1, static_cast<int>(std::round(bgr.rows * scale)));
    const int left = (kPalm - width) / 2, top = (kPalm - height) / 2;
    // Keep the exact x/y scale after integer resize rounding.
    const float sx = static_cast<float>(bgr.cols) / width, sy = static_cast<float>(bgr.rows) / height;
    cv::Mat resized, padded;
    cv::resize(bgr, resized, {width, height});
    cv::copyMakeBorder(resized, padded, top, kPalm - height - top, left, kPalm - width - left,
                       cv::BORDER_CONSTANT, cv::Scalar::all(0));
    auto ex = palm_.create_extractor();
    Require(ex.input("input", Input(padded)) == 0, "Palm input failed");
    ncnn::Mat cls, reg;
    Require(ex.extract("scores", cls) == 0 && ex.extract("regressors", reg) == 0, "Palm inference failed");
    Require(cls.w == 1 && cls.h == static_cast<int>(anchors_.size()) && cls.elempack == 1 &&
            reg.w == 18 && reg.h == cls.h && reg.elempack == 1, "Unexpected palm output shape");
    std::vector<Palm> candidates;
    for (int i = 0; i < cls.h; ++i) {
      const float logit = cls.row(i)[0];
      if (!std::isfinite(logit) || logit < 0) continue;  // sigmoid >= 0.5
      const float* r = reg.row(i);
      bool finite = true;
      for (int j = 0; j < 18; ++j) finite = finite && std::isfinite(r[j]);
      if (!finite || r[2] <= 0 || r[3] <= 0) continue;
      Palm p;
      const auto anchor = anchors_[i];
      const float cx = (r[0] + anchor.x - left) * sx, cy = (r[1] + anchor.y - top) * sy;
      const float w = r[2] * sx, h = r[3] * sy;
      p.box = {cx - w / 2, cy - h / 2, w, h};
      p.confidence = 1.f / (1.f + std::exp(-std::min(logit, 100.f)));
      for (int j = 0; j < 7; ++j)
        p.keypoints[j] = {(r[4 + j * 2] + anchor.x - left) * sx, (r[5 + j * 2] + anchor.y - top) * sy};
      candidates.push_back(p);
    }
    std::sort(candidates.begin(), candidates.end(), [](const Palm& a, const Palm& b) { return a.confidence > b.confidence; });
    std::vector<Palm> palms;
    // MediaPipe's WEIGHTED NMS: merge boxes/keypoints that overlap the top box.
    while (!candidates.empty() && static_cast<int>(palms.size()) < max_hands_) {
      const Palm top = candidates.front();
      Palm merged{};
      merged.confidence = top.confidence;
      float total = 0;
      std::vector<Palm> remaining;
      for (const auto& p : candidates) {
        if (IoU(top.box, p.box) > 0.3f) {
          total += p.confidence;
          merged.box.x += p.confidence * p.box.x;
          merged.box.y += p.confidence * p.box.y;
          merged.box.width += p.confidence * p.box.width;
          merged.box.height += p.confidence * p.box.height;
          for (int j = 0; j < 7; ++j) merged.keypoints[j] += p.keypoints[j] * p.confidence;
        } else remaining.push_back(p);
      }
      // Degenerate tiny boxes may have an IoU clamped to zero even with self.
      if (total <= 0) { candidates.erase(candidates.begin()); continue; }
      merged.box.x /= total;
      merged.box.y /= total;
      merged.box.width /= total;
      merged.box.height /= total;
      for (auto& p : merged.keypoints) p *= 1.f / total;
      palms.push_back(merged);
      candidates = std::move(remaining);
    }
    std::vector<Hand> hands;
    for (const auto& palm : palms) {
      // Wrist (0) -> middle-finger base (2) defines the upright crop rotation.
      const auto direction = palm.keypoints[2] - palm.keypoints[0];
      const float angle = kPi / 2 - std::atan2(-direction.y, direction.x);
      const float c = std::cos(angle), s = std::sin(angle);
      const float side = 2.6f * std::max(palm.box.width, palm.box.height);
      if (side < 1 || side > 20 * std::max(bgr.cols, bgr.rows)) continue;
      const cv::Point2f center(palm.box.x + palm.box.width / 2 + 0.5f * palm.box.height * s,
                               palm.box.y + palm.box.height / 2 - 0.5f * palm.box.height * c);
      const cv::Point2f axis_x(c * side / 2, s * side / 2), axis_y(-s * side / 2, c * side / 2);
      const cv::Point2f src[] = {center - axis_x - axis_y, center + axis_x - axis_y, center - axis_x + axis_y};
      const cv::Point2f dst[] = {{0,0}, {kHand,0}, {0,kHand}};
      const cv::Mat affine = cv::getAffineTransform(src, dst);
      cv::Mat crop, inverse;
      cv::warpAffine(bgr, crop, affine, {kHand, kHand}, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
      cv::invertAffineTransform(affine, inverse);
      const auto result = Landmark(crop);
      if (result.presence < 0.5f) continue;
      Hand hand;
      hand.palm_confidence = palm.confidence;
      hand.presence = result.presence;
      hand.right_probability = result.right_probability;
      for (int i = 0; i < 21; ++i) {
        const float x = result.points[i * 3], y = result.points[i * 3 + 1];
        hand.points[i] = {
            static_cast<float>(inverse.at<double>(0,0) * x + inverse.at<double>(0,1) * y + inverse.at<double>(0,2)),
            static_cast<float>(inverse.at<double>(1,0) * x + inverse.at<double>(1,1) * y + inverse.at<double>(1,2)),
            result.points[i * 3 + 2] * side / (kHand * 0.4f)};
        // World coordinates rotate with the crop, without image translation/scale.
        hand.world_points[i] = {c * result.world[i*3] - s * result.world[i*3+1],
                                s * result.world[i*3] + c * result.world[i*3+1], result.world[i*3+2]};
      }
      hands.push_back(hand);
    }
    return hands;
  }

  void Check() {
    const cv::Mat blank(240, 320, CV_8UC3, cv::Scalar::all(0));
    Require(Detect(blank).empty(), "Unexpected hand on blank frame");
    const auto blank_hand = Landmark(cv::Mat(kHand, kHand, CV_8UC3, cv::Scalar::all(0)));
    Require(blank_hand.presence < 0.5f, "Unexpected hand presence on blank crop");
    std::cout << "Model OK: official palm + four landmark outputs verified on CPU.\n";
  }

 private:
  template <std::size_t N>
  std::array<float, N> Extract(ncnn::Extractor& ex, const char* name) {
    ncnn::Mat tensor;
    Require(ex.extract(name, tensor) == 0, std::string("Cannot extract ") + name);
    Require(tensor.w * tensor.h * tensor.d * tensor.c == static_cast<int>(N) &&
            tensor.elempack == 1 && tensor.elemsize == sizeof(float), std::string("Unexpected output shape: ") + name);
    auto flat = tensor.reshape(static_cast<int>(N));
    Require(!flat.empty(), std::string("Cannot flatten ") + name);
    std::array<float, N> output;
    const float* data = flat;
    for (std::size_t i = 0; i < N; ++i) {
      Require(std::isfinite(data[i]), std::string("Non-finite output: ") + name);
      output[i] = data[i];
    }
    return output;
  }

  LandmarkResult Landmark(const cv::Mat& crop) {
    auto ex = landmark_.create_extractor();
    Require(ex.input("input", Input(crop)) == 0, "Landmark input failed");
    LandmarkResult result;
    result.points = Extract<63>(ex, "landmarks");
    result.presence = Extract<1>(ex, "presence")[0];
    result.right_probability = Extract<1>(ex, "handedness")[0];
    result.world = Extract<63>(ex, "world_landmarks");
    Require(result.presence >= 0 && result.presence <= 1 && result.right_probability >= 0 &&
            result.right_probability <= 1, "Invalid model probability");
    return result;
  }
  ncnn::Net palm_, landmark_;
  std::vector<cv::Point2f> anchors_;
  int max_hands_;
};
}  // namespace hand_detector
