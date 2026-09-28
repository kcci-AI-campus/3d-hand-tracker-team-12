// H3D1 (master -> PC) round trip and byte layout, and UV2 (slave -> master) decoding.
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "pc_packet.hpp"
#include "uv_sender.hpp"

using namespace hand_master;

void Check(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
}

int main() {
    try {
        pc::Output output;
        output.sequence = 0x0a0b0c0du;
        output.time_s = 12345.678901234;
        output.latency_ms = 87.5f;
        output.cameras_seen = {3, 0};
        for (std::size_t i = 0; i < output.joints_cm.size(); ++i) output.joints_cm[i] = static_cast<float>(i) - 60.25f;
        const auto bytes = pc::encode(output);
        Check(bytes.size() == 532, "H3D1 is 532 bytes");
        Check(bytes[0] == 'H' && bytes[1] == '3' && bytes[2] == 'D' && bytes[3] == '1' && bytes[4] == 1, "H3D1 header");
        Check(bytes[8] == 0x0a && bytes[11] == 0x0d && bytes[24] == 3 && bytes[25] == 0, "H3D1 big-endian fields");
        const auto back = pc::decode(bytes.data(), bytes.size());
        Check(back && back->sequence == output.sequence && back->time_s == output.time_s && back->latency_ms == 87.5f &&
              back->cameras_seen == output.cameras_seen && back->joints_cm == output.joints_cm, "H3D1 round trip");
        Check(!pc::decode(bytes.data(), 531), "H3D1 size check");

        uv_stream::Frame frame;
        frame.camera = 1;
        frame.sequence = 7;
        frame.capture_to_send_us = 61000;
        frame.uv.fill(-1.f);
        frame.uv[0] = .25f;
        const auto packet = uv_stream::Encode(frame);
        const auto decoded = uv_stream::Decode(packet.data(), packet.size());
        Check(decoded && decoded->camera == 1 && decoded->sequence == 7 && decoded->capture_to_send_us == 61000 &&
              decoded->uv[0] == .25f && decoded->uv[1] == -1.f, "UV2 decode");
        std::cout << "H3D1/UV2 packets OK\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
