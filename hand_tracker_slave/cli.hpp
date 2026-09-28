#pragma once
#include "uv_sender.hpp"
#include <utility>

inline std::pair<std::string, int> PromptDestination(std::istream& input, std::ostream& output) {
  std::string host, text;
  for (;;) {
    output << "Master IPv4: " << std::flush;
    if (!std::getline(input, host)) throw std::runtime_error("Master IP input cancelled");
    const auto first = host.find_first_not_of(" \t\r");
    host = first == std::string::npos ? "" : host.substr(first, host.find_last_not_of(" \t\r")-first+1);
    in_addr address{};
    if (inet_pton(AF_INET, host.c_str(), &address) == 1 &&
        ntohl(address.s_addr) != 0 && ntohl(address.s_addr) < 0xe0000000U) break;
    output << "Enter a unicast IPv4 address, e.g. 192.168.0.10.\n";
  }
  int port = 5001;
  for (;;) {
    output << "Master UDP port [5001]: " << std::flush;
    if (!std::getline(input, text)) throw std::runtime_error("Port input cancelled");
    if (text.empty() || text == "\r") break;
    try {
      std::size_t end;
      port = std::stoi(text, &end);
      if (text.find_first_not_of(" \t\r", end) == std::string::npos && port >= 1 && port <= 65535) break;
    } catch (const std::exception&) {}
    port = 5001;
    output << "Enter a port from 1 to 65535.\n";
  }
  return {host, port};
}
