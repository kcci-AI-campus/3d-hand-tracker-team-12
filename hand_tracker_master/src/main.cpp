// Master Pi app: two slave Pis' UV2 packets + this Pi's own camera -> HandDirect-wrist (default) or
// HandLiteV3 (--arch litev3) -> 3D joints to the PC.
//
//   slave cam 0 --UV2/UDP--> receiver thread ┐
//   slave cam 1 --UV2/UDP--> receiver thread ├─> inbox -> runtime thread: features, model push;
//   own camera (cam 2) -> detector thread ───┘            on every own frame: query -> H3D1/UDP -> PC
//
// Times are this Pi's steady clock in seconds. A slave frame's capture time is its receive time
// minus the capture->send time the slave measured and a small network latency, so the Pis need
// no clock sync. Outputs follow the own camera's frames, like the training data (query_sync_camera3).
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "cli.hpp"            // hand_tracker_slave: PromptDestination
#include "hand_detector.hpp"  // hand_tracker_slave: the same landmark models as the slaves
#include "uv_sender.hpp"      // hand_tracker_slave: UV2 decode, CoordinatesFrom

#include "direct_runtime.hpp"
#include "litev3_runtime.hpp"
#include "ncnn_graphs.hpp"
#include "pc_packet.hpp"
#include "rig.hpp"
#include "udp.hpp"

namespace fs = std::filesystem;
using namespace hand_master;
using Clock = std::chrono::steady_clock;

namespace {

volatile std::sig_atomic_t stopped = 0;
void Stop(int) { stopped = 1; }

constexpr int kOwnCamera = 2;

struct Options {
    int camera = 0, fps = 15, hands = 2, threads = 2, model_threads = 1;
    std::array<int, 2> slave_ports{{5001, 5002}};
    double net_latency_ms = 1.;
    fs::path models, model_dir, rig;
    std::string pc, image, arch = "direct";
    int pc_port = 6000;
    bool mirrored = false, fp16 = true, check = false;
};

double Seconds(Clock::time_point t) { return std::chrono::duration<double>(t.time_since_epoch()).count(); }
double Now() { return Seconds(Clock::now()); }

void Require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

int Integer(const std::string& text) {
    std::size_t end;
    const int value = std::stoi(text, &end);
    Require(end == text.size(), "Invalid integer: " + text);
    return value;
}

fs::path ExecutableDirectory(const char* argv0) {
#ifdef __linux__
    return fs::canonical("/proc/self/exe").parent_path();
#else
    return fs::absolute(fs::u8path(argv0)).parent_path();
#endif
}

// One camera frame's detections on the way to the runtime thread.
struct Incoming {
    int camera;
    double capture, arrival;
    uv_stream::Coordinates uv;
};

class Inbox {
public:
    void push(Incoming item) {
        { std::lock_guard<std::mutex> lock(mutex_); items_.push_back(item); }
        ready_.notify_one();
    }
    bool pop(Incoming& item, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!ready_.wait_for(lock, timeout, [&] { return !items_.empty(); })) return false;
        item = items_.front();
        items_.pop_front();
        return true;
    }
private:
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<Incoming> items_;
};

struct CameraStats {
    std::atomic<std::uint64_t> frames{0}, bad{0}, wrong_camera{0}, lost{0};
    std::atomic<double> age_ms{0.}, last_arrival{0.};
};

// UV2 packets of one slave camera on its own port.
void ReceiveSlave(int camera, int port, double net_latency_s, Inbox& inbox, CameraStats& stats,
                  std::atomic<bool>& failed) {
    try {
        udp::Socket socket;
        socket.bind(port, 200);
        std::array<unsigned char, 2048> buffer;
        bool has_sequence = false;
        std::uint32_t last_sequence = 0;
        while (!stopped) {
            const std::size_t size = socket.receive(buffer.data(), buffer.size());
            if (!size) continue;
            const double arrival = Now();
            const auto frame = uv_stream::Decode(buffer.data(), size);
            if (!frame) { ++stats.bad; continue; }
            if (frame->camera != camera) { ++stats.wrong_camera; continue; }
            if (has_sequence) {
                const std::uint32_t step = frame->sequence - last_sequence;  // wraps
                if (step == 0 || step > 0x80000000u) { ++stats.bad; continue; }  // duplicate or reordered
                stats.lost += step - 1;
            }
            has_sequence = true;
            last_sequence = frame->sequence;
            const double age = frame->capture_to_send_us * 1e-6 + net_latency_s;
            stats.age_ms = age * 1e3;
            stats.last_arrival = arrival;
            ++stats.frames;
            inbox.push({camera, arrival - age, arrival, frame->uv});
        }
    } catch (const std::exception& error) {
        std::cerr << "Slave camera " << camera << " receiver: " << error.what() << '\n';
        failed = true;
        stopped = 1;
    }
}

// This Pi's own camera (or a still image repeated, for tests without a camera).
void RunOwnCamera(const Options& options, Inbox& inbox, CameraStats& stats, std::atomic<double>& infer_ms,
                  std::atomic<bool>& failed) {
    try {
        hand_detector::HandDetector detector(options.models, options.hands, options.threads);
        cv::VideoCapture camera;
        cv::Mat still;
        if (!options.image.empty()) {
            still = cv::imread(options.image);
            Require(!still.empty(), "Cannot read image: " + options.image);
        } else {
#ifdef __linux__
            camera.open(options.camera, cv::CAP_V4L2);
#else
            camera.open(options.camera);
#endif
            Require(camera.isOpened(), "Cannot open USB camera " + std::to_string(options.camera));
            camera.set(cv::CAP_PROP_FRAME_WIDTH, 320);
            camera.set(cv::CAP_PROP_FRAME_HEIGHT, 240);
            camera.set(cv::CAP_PROP_FPS, options.fps);
            camera.set(cv::CAP_PROP_BUFFERSIZE, 1);
        }
        while (!stopped) {
            const auto start = Clock::now();
            cv::Mat raw, frame;
            if (still.empty()) {
                if (!camera.read(raw) || raw.empty()) {
                    if (stopped) break;
                    throw std::runtime_error("Camera read failed");
                }
            } else {
                raw = still;
            }
            const double capture = Now();  // same convention as the slaves: frame handed over
            cv::resize(raw, frame, {320, 240});
            const auto infer_start = Clock::now();
            const auto hands = detector.Detect(frame);
            infer_ms = std::chrono::duration<double, std::milli>(Clock::now() - infer_start).count();
            const double arrival = Now();  // local keypoints ready: the training data's camera-3 arrival
            stats.age_ms = (arrival - capture) * 1e3;
            stats.last_arrival = arrival;
            ++stats.frames;
            inbox.push({kOwnCamera, capture, arrival, uv_stream::CoordinatesFrom(hands, frame.cols, frame.rows, options.mirrored)});
            std::this_thread::sleep_until(start + std::chrono::duration<double>(1. / options.fps));
        }
    } catch (const std::exception& error) {
        std::cerr << "Own camera: " << error.what() << '\n';
        failed = true;
        stopped = 1;
    }
}

std::string Fixed(double value, int digits) {
    std::ostringstream text;
    text << std::fixed << std::setprecision(digits) << value;
    return text.str();
}

// The chosen model's runtime on its ncnn export.
std::unique_ptr<PoseRuntime> MakeRuntime(const Options& options) {
    const std::string dir = options.model_dir.u8string();
    if (options.arch == "litev3") {
        const LiteConfig config = load_lite_config(dir);
        return std::make_unique<LiteV3Runtime>(config, std::make_shared<LiteNcnnGraphs>(dir, config, options.model_threads,
                                                                                         options.fp16));
    }
    const Config config = load_config(dir);
    return std::make_unique<Runtime>(config, std::make_shared<NcnnGraphs>(dir, config, options.model_threads, options.fp16));
}

// Load the models and run them once on synthetic frames (no camera, no network).
int Check(const Options& options, PoseRuntime& runtime, const Rig& rig) {
    hand_detector::HandDetector(options.models, options.hands, options.threads).Check();
    uv_stream::Coordinates uv;
    for (int j = 0; j < 42; ++j) { uv[j * 2] = .4f + .005f * (j % 21); uv[j * 2 + 1] = .5f - .004f * (j % 21); }
    Features features;
    Valid valid;
    double t = Now();
    for (int camera = 0; camera < kCameras; ++camera) {
        make_features(rig, camera, uv, .05, features, valid);
        runtime.push(camera, features, valid, t - .05, t);
    }
    const auto start = Clock::now();
    const Pose pose = runtime.query(t);
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    for (float value : pose) Require(std::isfinite(value), "Non-finite model output");
    std::cout << "Model OK: " << (options.arch == "litev3" ? "HandLiteV3" : "HandDirect") << " query " << Fixed(ms, 1)
              << " ms, left wrist ("
              << Fixed(pose[0] * rig.world_unit_cm, 1) << ", " << Fixed(pose[1] * rig.world_unit_cm, 1) << ", "
              << Fixed(pose[2] * rig.world_unit_cm, 1) << ") cm\n";
    return 0;
}

int Run(const Options& options) {
    cv::setNumThreads(1);  // Avoid OpenCV and ncnn competing for CPU threads.
    const Rig rig = load_rig(options.rig.u8string());
    const auto model = MakeRuntime(options);
    PoseRuntime& runtime = *model;
    if (options.check) return Check(options, runtime, rig);

    udp::Socket pc;
    pc.connect_to(options.pc, options.pc_port);
    Inbox inbox;
    std::array<CameraStats, kCameras> stats;
    std::atomic<double> infer_ms{0.};
    std::atomic<bool> failed{false};
    std::vector<std::thread> threads;
    for (int camera = 0; camera < 2; ++camera)
        threads.emplace_back(ReceiveSlave, camera, options.slave_ports[camera], options.net_latency_ms * 1e-3,
                             std::ref(inbox), std::ref(stats[camera]), std::ref(failed));
    threads.emplace_back(RunOwnCamera, std::cref(options), std::ref(inbox), std::ref(stats[kOwnCamera]),
                         std::ref(infer_ms), std::ref(failed));
    std::cout << "Model: " << (options.arch == "litev3" ? "HandLiteV3" : "HandDirect-wrist") << " (" << options.model_dir.u8string()
              << ")\nSlaves: cam0 UDP " << options.slave_ports[0] << ", cam1 UDP " << options.slave_ports[1]
              << "  ->  PC " << options.pc << ':' << options.pc_port << " (H3D1, " << pc::kPacketSize
              << " bytes)\nNo preview. Ctrl+C to exit.\n" << std::flush;

    std::uint32_t sequence = 0;
    std::uint64_t outputs = 0, sent = 0, rejected = 0;
    std::array<std::uint64_t, kCameras> reported{};
    double query_ms = 0., latency_ms = 0.;
    auto last_report = Clock::now();
    std::uint64_t outputs_at_report = 0;
    Features features;
    Valid valid;
    Incoming item;
    while (!stopped) {
        if (inbox.pop(item, std::chrono::milliseconds(200))) {
            // Receive threads stamp arrivals independently; the runtime needs them in order.
            const double arrival = std::max(item.arrival, runtime.last_arrival());
            make_features(rig, item.camera, item.uv, arrival - item.capture, features, valid);
            if (!runtime.push(item.camera, features, valid, item.capture, arrival)) ++rejected;  // stale capture
            if (item.camera == kOwnCamera) {
                const double time = std::max(Now(), runtime.last_arrival());
                const auto start = Clock::now();
                const Pose pose = runtime.query(time);
                query_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                pc::Output output;
                output.sequence = sequence++;
                output.time_s = time;
                output.latency_ms = static_cast<float>((time - item.capture) * 1e3);
                latency_ms = output.latency_ms;
                const auto& seen = runtime.seen_by_cameras();
                output.cameras_seen = {static_cast<std::uint8_t>(seen[0]), static_cast<std::uint8_t>(seen[1])};
                if (const auto in_view = runtime.in_view_probability()) {
                    output.has_in_view = true;
                    output.in_view = *in_view;
                }
                for (std::size_t i = 0; i < pose.size(); ++i)
                    output.joints_cm[i] = static_cast<float>(pose[i] * rig.world_unit_cm);
                const auto bytes = pc::encode(output);
                sent += pc.send(bytes.data(), bytes.size());
                ++outputs;
            }
        }
        const auto now = Clock::now();
        if (now - last_report >= std::chrono::seconds(1)) {
            const double span = std::chrono::duration<double>(now - last_report).count();
            std::ostringstream line;
            line << "OUT " << Fixed((outputs - outputs_at_report) / span, 1) << " fps  latency " << Fixed(latency_ms, 0)
                 << "ms  query " << Fixed(query_ms, 1) << "ms  own infer " << Fixed(infer_ms, 0) << "ms  seen L/R "
                 << runtime.seen_by_cameras()[0] << '/' << runtime.seen_by_cameras()[1];
            if (const auto in_view = runtime.in_view_probability())
                line << "  in-view L/R " << Fixed((*in_view)[0], 2) << '/' << Fixed((*in_view)[1], 2);
            for (int c = 0; c < kCameras; ++c) {
                const auto frames = stats[c].frames.load();
                const bool live = Now() - stats[c].last_arrival < 1.;
                line << " | cam" << c << ' ' << Fixed((frames - reported[c]) / span, 1) << "fps age "
                     << Fixed(stats[c].age_ms, 0) << "ms" << (live ? "" : " (no data)");
                if (c < 2 && (stats[c].bad || stats[c].wrong_camera || stats[c].lost))
                    line << " bad " << stats[c].bad << " wrong-id " << stats[c].wrong_camera << " lost " << stats[c].lost;
                reported[c] = frames;
            }
            line << " | TX " << sent << " stale " << rejected;
            std::cout << line.str() << '\n' << std::flush;
            last_report = now;
            outputs_at_report = outputs;
        }
    }
    stopped = 1;
    for (auto& thread : threads) thread.join();
    std::cout << "Stopped. Outputs " << outputs << ", sent " << sent << std::endl;
    return failed ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, Stop);
    std::signal(SIGTERM, Stop);
    Options options;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--help" || key == "-h") {
                std::cout <<
                    "Usage: hand_tracker_master [--camera N] [--fps 1..60] [--hands 1|2] [--threads 1..64]\n"
                    "                           [--model-threads 1..8] [--no-fp16] [--slave-ports P0,P1]\n"
                    "                           [--net-latency-ms MS] [--models DIR] [--model DIR] [--rig FILE]\n"
                    "                           [--mirrored] [--image FILE] [--check-model] [--arch direct|litev3]\n"
                    "Defaults: own USB camera 0 (rig camera 2), 320x240, 15 FPS, slaves cam0 on UDP 5001 and\n"
                    "cam1 on 5002, network latency 1 ms, HandDirect-wrist in models/hand_direct_wrist\n"
                    "(--arch litev3: HandLiteV3 in models/hand_litev3).\n"
                    "PC IP and UDP port are prompted at startup. Outputs: H3D1 packets (see README). No GUI.\n";
                return 0;
            }
            if (key == "--check-model") { options.check = true; continue; }
            if (key == "--mirrored") { options.mirrored = true; continue; }
            if (key == "--no-fp16") { options.fp16 = false; continue; }
            Require(++i < argc, "Missing value for " + key);
            const std::string value = argv[i];
            if (key == "--camera") options.camera = Integer(value);
            else if (key == "--fps") options.fps = Integer(value);
            else if (key == "--hands") options.hands = Integer(value);
            else if (key == "--threads") options.threads = Integer(value);
            else if (key == "--model-threads") options.model_threads = Integer(value);
            else if (key == "--net-latency-ms") options.net_latency_ms = std::stod(value);
            else if (key == "--models") options.models = fs::u8path(value);
            else if (key == "--model") options.model_dir = fs::u8path(value);
            else if (key == "--rig") options.rig = fs::u8path(value);
            else if (key == "--image") options.image = value;
            else if (key == "--arch") options.arch = value;
            else if (key == "--slave-ports") {
                const auto comma = value.find(',');
                Require(comma != std::string::npos, "--slave-ports needs two ports, e.g. 5001,5002");
                options.slave_ports = {Integer(value.substr(0, comma)), Integer(value.substr(comma + 1))};
            } else throw std::runtime_error("Unknown option: " + key);
        }
        Require(options.camera >= 0 && options.fps >= 1 && options.fps <= 60 && options.hands >= 1 && options.hands <= 2 &&
                options.threads >= 1 && options.threads <= 64 && options.model_threads >= 1 && options.model_threads <= 8 &&
                options.net_latency_ms >= 0 && options.net_latency_ms < 1000,
                "camera >=0; fps 1..60; hands 1|2; threads 1..64; model-threads 1..8; net-latency-ms 0..1000");
        for (int port : options.slave_ports) Require(port >= 1 && port <= 65535, "slave ports must be 1..65535");
        Require(options.slave_ports[0] != options.slave_ports[1], "the two slave ports must differ");
        if (options.models.empty()) options.models = ExecutableDirectory(argv[0]) / "models";
        Require(options.arch == "direct" || options.arch == "litev3", "--arch must be direct or litev3");
        if (options.model_dir.empty())
            options.model_dir = options.models / (options.arch == "litev3" ? "hand_litev3" : "hand_direct_wrist");
        if (options.rig.empty()) options.rig = options.models / "rig.json";
        if (!options.check) {
            const auto destination = PromptDestination(std::cin, std::cout, "PC", 6000);
            options.pc = destination.first;
            options.pc_port = destination.second;
        }
        return Run(options);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
