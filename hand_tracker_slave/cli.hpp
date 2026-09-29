#pragma once
#include "uv_sender.hpp"
#include <utility>

// Destination IPv4 and UDP port; label names it in the prompts (the master app reuses this for the PC).
inline std::pair<std::string, int> PromptDestination(std::istream& input, std::ostream& output,
                                                     const std::string& label = "Master", int default_port = 5001) {
  std::string host, text;
  for (;;) {
    output << label << " IPv4: " << std::flush;
    if (!std::getline(input, host)) throw std::runtime_error(label + " IP input cancelled");
    const auto first = host.find_first_not_of(" \t\r");
    host = first == std::string::npos ? "" : host.substr(first, host.find_last_not_of(" \t\r")-first+1);
    in_addr address{};
    if (inet_pton(AF_INET, host.c_str(), &address) == 1 &&
        ntohl(address.s_addr) != 0 && ntohl(address.s_addr) < 0xe0000000U) break;
    output << "Enter a unicast IPv4 address, e.g. 192.168.0.10.\n";
  }
  int port = default_port;
  for (;;) {
    output << label << " UDP port [" << default_port << "]: " << std::flush;
    if (!std::getline(input, text)) throw std::runtime_error("Port input cancelled");
    if (text.empty() || text == "\r") break;
    try {
      std::size_t end;
      port = std::stoi(text, &end);
      if (text.find_first_not_of(" \t\r", end) == std::string::npos && port >= 1 && port <= 65535) break;
    } catch (const std::exception&) {}
    port = default_port;
    output << "Enter a port from 1 to 65535.\n";
  }
  return {host, port};
}

// Rig camera id of this slave: 0 or 1 (the master Pi's own camera is 2). It must match where the
// camera is installed in the rig (hand_tracker_master/models/rig.json); no default on purpose.
inline int PromptCameraId(std::istream& input, std::ostream& output) {
  std::string text;
  for (;;) {
    output << "Rig camera id (0 or 1): " << std::flush;
    if (!std::getline(input, text)) throw std::runtime_error("Camera id input cancelled");
    const auto first = text.find_first_not_of(" \t\r");
    text = first == std::string::npos ? "" : text.substr(first, text.find_last_not_of(" \t\r")-first+1);
    if (text == "0" || text == "1") return text[0] - '0';
    output << "Enter 0 or 1 (see hand_tracker_master/README.md for the camera positions).\n";
  }
}
