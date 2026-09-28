// Slave camera app: USB camera -> hand landmarks (hand_detector.hpp) -> UV2 UDP to the master Pi
// (hand_tracker_master). Models converted directly from Google's official hand_landmarker.task.
// See MODEL_STRUCTURE.md and conversion/parity_report.json for the contract.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <net.h>
#include <opencv2/core.hpp>
#include "cli.hpp"
#include "hand_detector.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using hand_detector::HandDetector;
using hand_detector::Require;
volatile std::sig_atomic_t stopped = 0;
void Stop(int) { stopped = 1; }

struct Options {
  int camera = 0, fps = 15, hands = 2, threads = 2, expect_hands = -1;
  fs::path models;
  std::string image, master;
  int port = 5001, camera_id = 0;
  bool check = false, mirrored = false, send_image = false;
};

fs::path ExecutableDirectory() {
  return fs::canonical("/proc/self/exe").parent_path();
}

int Integer(const std::string& text) {
  std::size_t end;
  const int value = std::stoi(text, &end);
  Require(end == text.size(), "Invalid integer: " + text);
  return value;
}

std::uint32_t Micros(Clock::duration duration) {
  const auto us = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
  return static_cast<std::uint32_t>(std::clamp<long long>(us, 0, 0xffffffffLL));
}

int Run(const Options& options) {
  cv::setNumThreads(1);  // Avoid OpenCV and ncnn competing for CPU threads.
  uv_stream::Sender sender(options.master, options.port);
  HandDetector detector(options.models, options.hands, options.threads);
  if (options.check) { detector.Check(); return 0; }
  std::uint32_t sequence = 0;
  if (!options.image.empty()) {
    auto image = cv::imread(options.image);
    Require(!image.empty(), "Cannot read image: " + options.image);
    const auto start = Clock::now();  // the image's "capture"
    const auto hands = detector.Detect(image);
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    std::cout << "Hands: " << hands.size() << "  Infer: " << ms << "ms\n";
    for (const auto& hand : hands) {
      std::cout << "Palm confidence: " << hand.palm_confidence << "  Presence: " << hand.presence
                << "  Model right probability: " << hand.right_probability << "\n";
      for (std::size_t i = 0; i < hand.points.size(); ++i)
        std::cout << i << ": " << hand.points[i] << '\n';
      for (std::size_t i = 0; i < hand.world_points.size(); ++i)
        std::cout << "world " << i << ": " << hand.world_points[i] << '\n';
    }
    uv_stream::Frame packet{options.camera_id, sequence++, 0, Micros(Clock::now() - start),
                            uv_stream::CoordinatesFrom(hands, image.cols, image.rows, options.mirrored)};
    packet.capture_to_send_us = Micros(Clock::now() - start);
    sender.Send(packet);
    if (options.send_image) std::cout << sender.Status() << std::endl;
    Require(options.expect_hands < 0 || static_cast<int>(hands.size()) == options.expect_hands, "Unexpected number of hands");
    return 0;
  }
  cv::VideoCapture camera;
  camera.open(options.camera, cv::CAP_V4L2);
  Require(camera.isOpened(), "Cannot open USB camera " + std::to_string(options.camera));
  camera.set(cv::CAP_PROP_FRAME_WIDTH, 320);
  camera.set(cv::CAP_PROP_FRAME_HEIGHT, 240);
  camera.set(cv::CAP_PROP_FPS, options.fps);
  camera.set(cv::CAP_PROP_BUFFERSIZE, 1);
  std::deque<Clock::time_point> displayed;
  auto last_report = Clock::now();
  std::cout << "Camera ready (rig camera " << options.camera_id << "). No preview. Ctrl+C to exit.\n" << std::flush;
  while (!stopped) {
    const auto start = Clock::now();
    cv::Mat raw, frame;
    if (!camera.read(raw) || raw.empty()) {
      if (stopped) break;
      throw std::runtime_error("Camera read failed");
    }
    // Capture time: when the frame is handed over (the exposure is slightly earlier; the master's
    // own camera uses the same convention, so the rig stays consistent).
    const auto captured = Clock::now();
    cv::resize(raw, frame, {320,240});
    const auto infer_start = Clock::now();
    const auto hands = detector.Detect(frame);
    const auto infer_end = Clock::now();
    uv_stream::Frame packet{options.camera_id, sequence++, 0, Micros(infer_end - infer_start),
                            uv_stream::CoordinatesFrom(hands, frame.cols, frame.rows, options.mirrored)};
    packet.capture_to_send_us = Micros(Clock::now() - captured);
    sender.Send(packet);
    const auto now = Clock::now();
    displayed.push_back(now);
    while (displayed.size() > 120 || (displayed.size() > 2 && now - displayed.front() > std::chrono::seconds(1))) displayed.pop_front();
    const double span = std::chrono::duration<double>(now - displayed.front()).count();
    const std::string fps = span > 0 ? cv::format("%.1f", (displayed.size() - 1) / span) : "--";
    const std::string label = "FPS: " + fps + "  Hands: " + std::to_string(hands.size()) +
        cv::format("  Infer: %.0fms", packet.infer_us / 1000.) + cv::format("  Age: %.0fms", packet.capture_to_send_us / 1000.);
    if (now - last_report >= std::chrono::seconds(1)) {
      std::cout << "cam" << options.camera_id << " -> " << options.master << ":" << options.port << "  " << label
                << "  " << sender.Status() << '\n' << std::flush;
      last_report = now;
    }
    std::this_thread::sleep_until(start + std::chrono::duration<double>(1. / options.fps));
  }
  std::cout << "Stopped. " << sender.Status() << std::endl;
  return 0;
}

int main(int argc, char** argv) {
  std::signal(SIGINT, Stop);
  std::signal(SIGTERM, Stop);
  int status = 0;
  Options options;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string key = argv[i];
      if (key == "--help" || key == "-h") {
        std::cout << "Usage: hand_tracker_slave [--camera N] [--fps 1..60] [--hands 1|2] [--threads 1..64]\n"
                     "                    [--models DIR] [--check-model]\n"
                     "                    [--image FILE [--expect-hands N] [--send-uv]] [--mirrored]\n"
                     "Defaults: USB 0, 320x240, 15 FPS, 2 hands, 2 CPU threads.\n"
                     "Master IP, UDP port and rig camera id (0|1) are prompted at startup. No GUI.\n"
                     "UV2: 356-byte packets (20-byte header + 84 floats); default camera is unmirrored.\n";
        return 0;
      }
      if (key == "--check-model") { options.check = true; continue; }
      if (key == "--headless" || key == "--print-fps") continue; // Legacy no-op aliases.
      if (key == "--send-uv") { options.send_image = true; continue; }
      if (key == "--mirrored") { options.mirrored = true; continue; }
      Require(key == "--camera" || key == "--fps" || key == "--hands" || key == "--threads" ||
              key == "--models" || key == "--image" || key == "--expect-hands", "Unknown option: " + key);
      Require(++i < argc, "Missing value for " + key);
      const std::string value = argv[i];
      if (key == "--camera") options.camera = Integer(value);
      else if (key == "--fps") options.fps = Integer(value);
      else if (key == "--hands") options.hands = Integer(value);
      else if (key == "--threads") options.threads = Integer(value);
      else if (key == "--models") options.models = fs::u8path(value);
      else if (key == "--image") options.image = value;
      else options.expect_hands = Integer(value);
    }
    Require(options.camera >= 0 && options.fps >= 1 && options.fps <= 60 && options.hands >= 1 && options.hands <= 2 &&
            options.threads >= 1 && options.threads <= 64 && options.expect_hands >= -1 && options.expect_hands <= 2,
            "camera >=0; fps 1..60; hands 1|2; threads 1..64; expect-hands 0..2");
    Require(options.expect_hands == -1 || !options.image.empty(), "--expect-hands requires --image");
    Require(!options.send_image || (!options.image.empty() && !options.check), "--send-uv requires --image without --check-model");
    if (!options.check && (options.image.empty() || options.send_image)) {
      const auto destination = PromptDestination(std::cin, std::cout);
      options.master = destination.first;
      options.port = destination.second;
      options.camera_id = PromptCameraId(std::cin, std::cout);
    }
    if (options.models.empty()) options.models = ExecutableDirectory() / "models";
    status = Run(options);
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    status = 1;
  }
  return status;
}
