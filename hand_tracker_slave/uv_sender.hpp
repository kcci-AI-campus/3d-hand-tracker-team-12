#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace uv_stream {
using Coordinates = std::array<float, 84>; // [left 21][right 21], interleaved u,v
using Packet = std::array<unsigned char, 336>;

template<class Hands>
Coordinates CoordinatesFrom(const Hands& hands, int width, int height, bool mirrored) {
  if (width <= 0 || height <= 0) throw std::invalid_argument("Invalid image size");
  Coordinates uv;
  uv.fill(-1.f);
  // Classification may report the same side twice; keep the most confident one.
  std::array<float, 2> best{{-1.f, -1.f}};
  for (const auto& hand : hands) {
    if (!std::isfinite(hand.right_probability) || !std::isfinite(hand.presence)) continue;
    const float right = mirrored ? hand.right_probability : 1.f - hand.right_probability;
    const int slot = right >= .5f ? 1 : 0;
    const float confidence = hand.presence * (slot ? right : 1.f - right);
    if (confidence <= best[slot]) continue;
    best[slot] = confidence;
    for (int j = 0; j < 21; ++j) {
      const auto& p = hand.points[j];
      const float u = p.x / width, v = p.y / height;
      const bool valid = std::isfinite(u) && std::isfinite(v) && u >= 0 && u <= 1 && v >= 0 && v <= 1;
      uv[slot*42+j*2] = valid ? u : -1.f;
      uv[slot*42+j*2+1] = valid ? v : -1.f;
    }
  }
  return uv;
}

inline Packet Encode(const Coordinates& uv) {
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "IEEE754 float32 required");
  Packet bytes{};
  for (std::size_t i = 0; i < uv.size(); ++i) {
    std::uint32_t bits;
    std::memcpy(&bits, &uv[i], 4);
    for (int k = 0; k < 4; ++k) bytes[i*4+k] = static_cast<unsigned char>(bits >> (24-8*k));
  }
  return bytes;
}

class Sender {
 public:
  Sender(const std::string& host, int port) {
    if (host.empty()) return; // Explicit local/check mode.
    if (port < 1 || port > 65535) throw std::invalid_argument("port must be 1..65535");
    address_.sin_family = AF_INET;
    address_.sin_port = htons(static_cast<unsigned short>(port));
    if (inet_pton(AF_INET, host.c_str(), &address_.sin_addr) != 1)
      throw std::invalid_argument("Master requires an IPv4 address");
#ifdef _WIN32
    WSADATA data;
    if (WSAStartup(MAKEWORD(2,2), &data)) throw std::runtime_error("WSAStartup failed");
    started_ = true;
#endif
    socket_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_ == invalid_) { Close(); throw std::runtime_error("Cannot create UDP socket"); }
#ifdef _WIN32
    u_long nonblocking = 1;
    if (ioctlsocket(socket_, FIONBIO, &nonblocking)) {
      Close(); throw std::runtime_error("Cannot set nonblocking UDP");
    }
#endif
    std::cout << "UV42 UDP -> " << host << ':' << port << " (336 bytes/frame)\n"
              << "TX counts local sends; UDP delivery is not acknowledged.\n" << std::flush;
  }
  ~Sender() { Close(); }
  Sender(const Sender&) = delete;
  Sender& operator=(const Sender&) = delete;
  void Send(const Coordinates& uv) {
    if (socket_ == invalid_) return;
    const auto bytes = Encode(uv);
#ifdef _WIN32
    const int flags = 0;
#else
    const int flags = MSG_DONTWAIT;
#endif
    const auto sent = sendto(socket_, reinterpret_cast<const char*>(bytes.data()),
        static_cast<int>(bytes.size()), flags, reinterpret_cast<const sockaddr*>(&address_), sizeof(address_));
    if (sent == static_cast<int>(bytes.size())) { ++sent_; last_error_ = 0; }
    else {
      ++errors_;
#ifdef _WIN32
      last_error_ = WSAGetLastError();
#else
      last_error_ = errno;
#endif
    }
  }
  std::uint64_t Sent() const { return sent_; }
  std::uint64_t Failed() const { return errors_; }
  int LastError() const { return last_error_; }
  std::string Status() const {
    return "TX: " + std::to_string(sent_) + "  Failed: " + std::to_string(errors_) +
           "  Bytes: " + std::to_string(sent_*336) + "  Last socket error: " + std::to_string(last_error_);
  }
 private:
  void Close() {
#ifdef _WIN32
    if (socket_ != invalid_) closesocket(socket_);
    if (started_) { WSACleanup(); started_ = false; }
#else
    if (socket_ != invalid_) close(socket_);
#endif
    socket_ = invalid_;
  }
#ifdef _WIN32
  using Socket = SOCKET;
  static constexpr Socket invalid_ = INVALID_SOCKET;
  bool started_ = false;
#else
  using Socket = int;
  static constexpr Socket invalid_ = -1;
#endif
  Socket socket_ = invalid_;
  sockaddr_in address_{};
  std::uint64_t errors_ = 0, sent_ = 0;
  int last_error_ = 0;
};
} // namespace uv_stream
