// Small UDP sockets (POSIX on the Pi; Winsock so the app and its tests also build on a PC).
#pragma once
#include <cerrno>
#include <cstdint>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

namespace hand_master::udp {

class Socket {
public:
    Socket() {
#ifdef _WIN32
        WSADATA data;
        if (WSAStartup(MAKEWORD(2, 2), &data)) throw std::runtime_error("WSAStartup failed");
        started_ = true;
#endif
        socket_ = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (socket_ == invalid_) { close(); throw std::runtime_error("Cannot create UDP socket"); }
    }
    ~Socket() { close(); }
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;

    // Receive on every interface at port; recv() gives up after timeout_ms (so threads can stop).
    void bind(int port, int timeout_ms) {
        if (port < 1 || port > 65535) throw std::invalid_argument("port must be 1..65535");
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_ANY);
        address.sin_port = htons(static_cast<unsigned short>(port));
        if (::bind(socket_, reinterpret_cast<const sockaddr*>(&address), sizeof(address)))
            throw std::runtime_error("Cannot bind UDP port " + std::to_string(port) + " (in use?)");
#ifdef _WIN32
        DWORD timeout = static_cast<DWORD>(timeout_ms);
#else
        timeval timeout{timeout_ms / 1000, (timeout_ms % 1000) * 1000};
#endif
        setsockopt(socket_, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout), sizeof(timeout));
    }

    // Bytes received (0 on timeout); throws on a real error.
    std::size_t receive(unsigned char* buffer, std::size_t capacity) {
        const auto got = ::recvfrom(socket_, reinterpret_cast<char*>(buffer), static_cast<int>(capacity), 0, nullptr, nullptr);
        if (got >= 0) return static_cast<std::size_t>(got);
#ifdef _WIN32
        const int error = WSAGetLastError();
        if (error == WSAETIMEDOUT || error == WSAEWOULDBLOCK || error == WSAEMSGSIZE) return 0;
#else
        const int error = errno;
        if (error == EAGAIN || error == EWOULDBLOCK || error == EINTR) return 0;
#endif
        throw std::runtime_error("UDP receive failed: error " + std::to_string(error));
    }

    void connect_to(const std::string& host, int port) {
        if (port < 1 || port > 65535) throw std::invalid_argument("port must be 1..65535");
        destination_.sin_family = AF_INET;
        destination_.sin_port = htons(static_cast<unsigned short>(port));
        if (inet_pton(AF_INET, host.c_str(), &destination_.sin_addr) != 1)
            throw std::invalid_argument("Destination must be an IPv4 address");
        has_destination_ = true;
    }

    // Non-blocking best effort; returns whether the OS accepted the datagram.
    bool send(const unsigned char* data, std::size_t size) {
        if (!has_destination_) return false;
#ifdef _WIN32
        const int flags = 0;
#else
        const int flags = MSG_DONTWAIT;
#endif
        return ::sendto(socket_, reinterpret_cast<const char*>(data), static_cast<int>(size), flags,
                        reinterpret_cast<const sockaddr*>(&destination_), sizeof(destination_)) == static_cast<int>(size);
    }

private:
    void close() {
#ifdef _WIN32
        if (socket_ != invalid_) closesocket(socket_);
        if (started_) { WSACleanup(); started_ = false; }
#else
        if (socket_ != invalid_) ::close(socket_);
#endif
        socket_ = invalid_;
    }
#ifdef _WIN32
    using Handle = SOCKET;
    static constexpr Handle invalid_ = INVALID_SOCKET;
    bool started_ = false;
#else
    using Handle = int;
    static constexpr Handle invalid_ = -1;
#endif
    Handle socket_ = invalid_;
    sockaddr_in destination_{};
    bool has_destination_ = false;
};

}  // namespace hand_master::udp
