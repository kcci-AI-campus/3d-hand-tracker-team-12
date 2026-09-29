#pragma once
// H3D1: one UDP datagram per model output, master Pi -> PC (tools/pc_receiver.py reads it).
//
//   offset size  field (big-endian)
//   0      4     magic "H3D1"
//   4      1     version (1)
//   5      1     flags: bit 0 = bytes 26, 27 hold the in-view probabilities (HandLiteV3)
//   6      2     reserved (0)
//   8      4     output sequence number
//   12     8     float64 query time, s (master steady clock; differences are meaningful)
//   20     4     float32 latency, ms: query time - capture of the master camera frame that triggered it
//   24     2     per hand (left, right): cameras whose recent frames saw it (0 = unseen, joints are a guess)
//   26     2     per hand: the model's probability that it is in some camera's view, x255 (flag bit 0)
//   28     504   126 float32: [left 21][right 21] joints x,y,z in cm, rig frame (Y up, origin where
//                 the three cameras look; see README)
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>

namespace hand_master::pc {

constexpr std::size_t kHeaderSize = 28, kPacketSize = kHeaderSize + 126 * 4;  // 532 bytes
constexpr std::uint8_t kVersion = 1;
constexpr unsigned char kMagic[4] = {'H', '3', 'D', '1'};
using Packet = std::array<unsigned char, kPacketSize>;

struct Output {
    std::uint32_t sequence = 0;
    double time_s = 0.;
    float latency_ms = 0.f;
    std::array<std::uint8_t, 2> cameras_seen{};
    bool has_in_view = false;
    std::array<float, 2> in_view{};   // 0..1, sent with 1/255 resolution
    std::array<float, 126> joints_cm{};
};

namespace detail {
inline void put32(unsigned char* out, std::uint32_t bits) {
    for (int k = 0; k < 4; ++k) out[k] = static_cast<unsigned char>(bits >> (24 - 8 * k));
}
inline std::uint32_t get32(const unsigned char* in) {
    return std::uint32_t(in[0]) << 24 | std::uint32_t(in[1]) << 16 | std::uint32_t(in[2]) << 8 | in[3];
}
inline std::uint32_t bits(float value) { std::uint32_t b; std::memcpy(&b, &value, 4); return b; }
inline float from_bits(std::uint32_t b) { float value; std::memcpy(&value, &b, 4); return value; }
}  // namespace detail

inline Packet encode(const Output& output) {
    static_assert(std::numeric_limits<float>::is_iec559 && sizeof(double) == 8, "IEEE754 required");
    Packet bytes{};
    std::memcpy(bytes.data(), kMagic, 4);
    bytes[4] = kVersion;
    bytes[5] = output.has_in_view ? 1 : 0;
    detail::put32(&bytes[8], output.sequence);
    std::uint64_t time_bits;
    std::memcpy(&time_bits, &output.time_s, 8);
    detail::put32(&bytes[12], static_cast<std::uint32_t>(time_bits >> 32));
    detail::put32(&bytes[16], static_cast<std::uint32_t>(time_bits));
    detail::put32(&bytes[20], detail::bits(output.latency_ms));
    bytes[24] = output.cameras_seen[0];
    bytes[25] = output.cameras_seen[1];
    for (int h = 0; h < 2; ++h) {
        const float p = output.has_in_view ? output.in_view[h] : 0.f;
        bytes[26 + h] = static_cast<unsigned char>(p <= 0.f ? 0 : p >= 1.f ? 255 : static_cast<int>(p * 255.f + .5f));
    }
    for (std::size_t i = 0; i < output.joints_cm.size(); ++i)
        detail::put32(&bytes[kHeaderSize + i * 4], detail::bits(output.joints_cm[i]));
    return bytes;
}

inline std::optional<Output> decode(const unsigned char* data, std::size_t size) {
    if (size != kPacketSize || std::memcmp(data, kMagic, 4) != 0 || data[4] != kVersion) return std::nullopt;
    Output output;
    output.sequence = detail::get32(&data[8]);
    const std::uint64_t time_bits = std::uint64_t(detail::get32(&data[12])) << 32 | detail::get32(&data[16]);
    std::memcpy(&output.time_s, &time_bits, 8);
    output.latency_ms = detail::from_bits(detail::get32(&data[20]));
    output.cameras_seen = {data[24], data[25]};
    output.has_in_view = (data[5] & 1) != 0;
    if (output.has_in_view) output.in_view = {data[26] / 255.f, data[27] / 255.f};
    for (std::size_t i = 0; i < output.joints_cm.size(); ++i)
        output.joints_cm[i] = detail::from_bits(detail::get32(&data[kHeaderSize + i * 4]));
    return output;
}

}  // namespace hand_master::pc
