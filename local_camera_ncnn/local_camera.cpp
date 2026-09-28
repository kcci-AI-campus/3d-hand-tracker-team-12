// CPU hand pipeline: palm detection -> rotated crop -> 21 landmarks.
// Model contract follows FeiGeChuanShu/ncnn-Android-mediapipe_hand.
// See THIRD_PARTY.md for model provenance and reference attribution.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <net.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
constexpr int kPalm = 192, kHand = 224;
constexpr float kPi = 3.14159265358979323846f;
volatile std::sig_atomic_t stopped = 0;
void Stop(int) { stopped = 1; }

struct Palm {
  cv::Rect2f box;
  std::array<cv::Point2f, 7> keypoints;
  float confidence;
};
struct Hand {
  std::array<cv::Point3f, 21> points;  // x/y image pixels; z image-pixel scale, not metres.
  float palm_confidence;
};
struct Options {
  int camera = 0, fps = 15, hands = 2, threads = 2, expect_hands = -1;
  fs::path models;
  std::string image, output;
  bool check = false;
};

fs::path ExecutableDirectory() {
#ifdef _WIN32
  std::vector<wchar_t> path(32768);
  const DWORD size = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
  if (!size || size >= path.size()) throw std::runtime_error("Cannot find executable path");
  return fs::path(std::wstring(path.data(), size)).parent_path();
#else
  return fs::canonical("/proc/self/exe").parent_path();
#endif
}

void Require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}

// fopen(path.string()) in ncnn does not reliably handle Unicode Windows paths.
void Load(ncnn::Net& net, const fs::path& base) {
  for (const auto& suffix : {".param", ".bin"}) {
    fs::path path = base;
    path += suffix;
#ifdef _WIN32
    FILE* file = _wfopen(path.c_str(), L"rb");
#else
    FILE* file = std::fopen(path.c_str(), "rb");
#endif
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
  explicit HandDetector(const Options& options) : max_hands_(options.hands) {
    for (auto* net : {&palm_, &landmark_}) {
      net->opt.num_threads = options.threads;
      net->opt.use_vulkan_compute = false;
    }
    // This older conversion's Squeeze -> Gemm head assumes unpacked channels.
    // Packed x86 activations otherwise produce 8 rows instead of 63 scalars.
    landmark_.opt.use_packing_layout = false;
    Load(palm_, options.models / "palm-lite-op");
    Load(landmark_, options.models / "hand_lite-op");
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
    Require(ex.extract("cls", cls) == 0 && ex.extract("reg", reg) == 0, "Palm inference failed");
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
      p.confidence = 1.f / (1.f + std::exp(-std::min(logit, 80.f)));
      for (int j = 0; j < 7; ++j)
        p.keypoints[j] = {(r[4 + j * 2] + anchor.x - left) * sx, (r[5 + j * 2] + anchor.y - top) * sy};
      candidates.push_back(p);
    }
    std::sort(candidates.begin(), candidates.end(), [](const Palm& a, const Palm& b) { return a.confidence > b.confidence; });
    std::vector<Palm> palms;
    for (const auto& p : candidates) {
      bool overlap = false;
      for (const auto& selected : palms) overlap = overlap || IoU(p.box, selected.box) > 0.3f;
      if (!overlap) palms.push_back(p);
      if (static_cast<int>(palms.size()) >= max_hands_) break;
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
      auto points = Landmark(crop);
      Hand hand;
      hand.palm_confidence = palm.confidence;
      for (int i = 0; i < 21; ++i) {
        const float x = points[i * 3], y = points[i * 3 + 1];
        hand.points[i] = {
            static_cast<float>(inverse.at<double>(0,0) * x + inverse.at<double>(0,1) * y + inverse.at<double>(0,2)),
            static_cast<float>(inverse.at<double>(1,0) * x + inverse.at<double>(1,1) * y + inverse.at<double>(1,2)),
            points[i * 3 + 2] * side / kHand};
      }
      hands.push_back(hand);
    }
    return hands;
  }

  void Check() {
    const cv::Mat blank(240, 320, CV_8UC3, cv::Scalar::all(0));
    Require(Detect(blank).empty(), "Unexpected hand on blank frame");
    Landmark(cv::Mat(kHand, kHand, CV_8UC3, cv::Scalar::all(0)));
    std::cout << "Model OK: palm AND landmark inference completed on CPU.\n";
  }

 private:
  std::array<float, 63> Landmark(const cv::Mat& crop) {
    auto ex = landmark_.create_extractor();
    Require(ex.input("input", Input(crop)) == 0, "Landmark input failed");
    ncnn::Mat points;
    Require(ex.extract("points", points) == 0, "Landmark inference failed");
    const std::string shape = "w=" + std::to_string(points.w) + " h=" + std::to_string(points.h) +
        " c=" + std::to_string(points.c) + " pack=" + std::to_string(points.elempack) + " bytes=" + std::to_string(points.elemsize);
    points = points.reshape(63);
    Require(!points.empty() && points.elempack == 1 && points.elemsize == sizeof(float), "Unexpected landmark output shape: " + shape);
    std::array<float, 63> output;
    const float* data = points;
    for (int i = 0; i < 63; ++i) {
      Require(std::isfinite(data[i]), "Non-finite landmark");
      output[i] = data[i];
    }
    // The bundled model's `score` is handedness, NOT hand-presence confidence.
    // Do not threshold it: that would discard one side's hands.
    return output;
  }
  ncnn::Net palm_, landmark_;
  std::vector<cv::Point2f> anchors_;
  int max_hands_;
};

void Draw(cv::Mat& frame, const std::vector<Hand>& hands) {
  constexpr int edges[][2] = {{0,1},{1,2},{2,3},{3,4},{0,5},{5,6},{6,7},{7,8},{5,9},{9,10},
    {10,11},{11,12},{9,13},{13,14},{14,15},{15,16},{13,17},{0,17},{17,18},{18,19},{19,20}};
  for (const auto& hand : hands) {
    std::array<cv::Point, 21> points;
    for (int i = 0; i < 21; ++i)
      points[i] = {cvRound(std::clamp(hand.points[i].x, 0.f, float(frame.cols - 1))),
                   cvRound(std::clamp(hand.points[i].y, 0.f, float(frame.rows - 1)))};
    for (const auto& edge : edges) cv::line(frame, points[edge[0]], points[edge[1]], {0,255,0}, 1);
    for (const auto& p : points) cv::circle(frame, p, 2, {0,0,255}, cv::FILLED);
  }
}

int Integer(const std::string& text) {
  std::size_t end;
  const int value = std::stoi(text, &end);
  Require(end == text.size(), "Invalid integer: " + text);
  return value;
}

int Run(const Options& options) {
  cv::setNumThreads(1);  // Avoid OpenCV and ncnn competing for CPU threads.
  HandDetector detector(options);
  if (options.check) { detector.Check(); return 0; }
  if (!options.image.empty()) {
    auto image = cv::imread(options.image);
    Require(!image.empty(), "Cannot read image: " + options.image);
    const auto start = Clock::now();
    const auto hands = detector.Detect(image);
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    std::cout << "Hands: " << hands.size() << "  Infer: " << ms << "ms\n";
    for (const auto& hand : hands) {
      std::cout << "Palm confidence: " << hand.palm_confidence << "\n";
      for (std::size_t i = 0; i < hand.points.size(); ++i)
        std::cout << i << ": " << hand.points[i] << '\n';
    }
    Draw(image, hands);
    if (!options.output.empty()) Require(cv::imwrite(options.output, image), "Cannot write output image");
    Require(options.expect_hands < 0 || static_cast<int>(hands.size()) == options.expect_hands, "Unexpected number of hands");
    return 0;
  }
  cv::VideoCapture camera;
#ifdef _WIN32
  camera.open(options.camera, cv::CAP_DSHOW);
#else
  camera.open(options.camera, cv::CAP_V4L2);
#endif
  Require(camera.isOpened(), "Cannot open USB camera " + std::to_string(options.camera));
  camera.set(cv::CAP_PROP_FRAME_WIDTH, 320);
  camera.set(cv::CAP_PROP_FRAME_HEIGHT, 240);
  camera.set(cv::CAP_PROP_FPS, options.fps);
  camera.set(cv::CAP_PROP_BUFFERSIZE, 1);
  constexpr char window[] = "Local Hand Landmarker - ncnn CPU";
  cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
  std::deque<Clock::time_point> displayed;
  std::cout << "Camera ready. Q / Esc / Ctrl+C to exit.\n";
  while (!stopped) {
    const auto start = Clock::now();
    cv::Mat raw, frame;
    if (!camera.read(raw) || raw.empty()) {
      if (stopped) break;
      throw std::runtime_error("Camera read failed");
    }
    cv::resize(raw, frame, {320,240});
    const auto infer_start = Clock::now();
    const auto hands = detector.Detect(frame);
    const double infer_ms = std::chrono::duration<double, std::milli>(Clock::now() - infer_start).count();
    Draw(frame, hands);
    const auto now = Clock::now();
    displayed.push_back(now);
    while (displayed.size() > 120 || (displayed.size() > 2 && now - displayed.front() > std::chrono::seconds(1))) displayed.pop_front();
    const double span = std::chrono::duration<double>(now - displayed.front()).count();
    const std::string fps = span > 0 ? cv::format("%.1f", (displayed.size() - 1) / span) : "--";
    const std::string label = "FPS: " + fps + "  Hands: " + std::to_string(hands.size()) + cv::format("  Infer: %.0fms", infer_ms);
    cv::rectangle(frame, {0,0}, {319,24}, {0,0,0}, cv::FILLED);
    cv::putText(frame, label, {7,17}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {255,255,255}, 1);
    cv::imshow(window, frame);
    const double spent = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    const int key = cv::waitKey(std::max(1, cvRound(1000. / options.fps - spent))) & 255;
    if (key == 27 || key == 'q' || key == 'Q' || cv::getWindowProperty(window, cv::WND_PROP_VISIBLE) < 1) break;
  }
  return 0;
}

int main(int argc, char** argv) {
  std::signal(SIGINT, Stop);
  std::signal(SIGTERM, Stop);
  int status = 0;
  try {
    Options options;
    for (int i = 1; i < argc; ++i) {
      const std::string key = argv[i];
      if (key == "--help" || key == "-h") {
        std::cout << "Usage: local_camera [--camera N] [--fps 1..60] [--hands 1|2] [--threads 1..64]\n"
                     "                    [--models DIR] [--check-model]\n"
                     "                    [--image FILE [--output FILE] [--expect-hands N]]\n"
                     "Defaults: USB 0, 320x240, 15 FPS, 2 hands, 2 CPU threads.\n";
        return 0;
      }
      if (key == "--check-model") { options.check = true; continue; }
      Require(key == "--camera" || key == "--fps" || key == "--hands" || key == "--threads" ||
              key == "--models" || key == "--image" || key == "--output" || key == "--expect-hands", "Unknown option: " + key);
      Require(++i < argc, "Missing value for " + key);
      const std::string value = argv[i];
      if (key == "--camera") options.camera = Integer(value);
      else if (key == "--fps") options.fps = Integer(value);
      else if (key == "--hands") options.hands = Integer(value);
      else if (key == "--threads") options.threads = Integer(value);
      else if (key == "--models") options.models = fs::u8path(value);
      else if (key == "--image") options.image = value;
      else if (key == "--output") options.output = value;
      else options.expect_hands = Integer(value);
    }
    Require(options.camera >= 0 && options.fps >= 1 && options.fps <= 60 && options.hands >= 1 && options.hands <= 2 &&
            options.threads >= 1 && options.threads <= 64 && options.expect_hands >= -1 && options.expect_hands <= 2,
            "camera >=0; fps 1..60; hands 1|2; threads 1..64; expect-hands 0..2");
    Require((options.output.empty() && options.expect_hands == -1) || !options.image.empty(), "--output/--expect-hands require --image");
    if (options.models.empty()) options.models = ExecutableDirectory() / "models";
    status = Run(options);
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    status = 1;
  }
  try { cv::destroyAllWindows(); } catch (const cv::Exception&) {}
  return status;
}
