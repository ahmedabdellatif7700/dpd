#include "stage/net.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <ctime>
#include <stdexcept>
#include <thread>

namespace arof {

namespace {

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error(what + ": " + std::strerror(errno)); }

// 1: all n bytes read; 0: end of stream before the first byte; throws if cut short
int read_all(int fd, void* buf, size_t n) {
    auto* p = static_cast<unsigned char*>(buf);
    size_t got = 0;
    while (got < n) {
        const ssize_t k = ::recv(fd, p + got, n - got, 0);
        if (k == 0) {
            if (got == 0) return 0;
            throw std::runtime_error("connection closed inside a frame");
        }
        if (k < 0) {
            if (errno == EINTR) continue;
            fail("recv");
        }
        got += static_cast<size_t>(k);
    }
    return 1;
}

void write_all(int fd, const void* buf, size_t n) {
    const auto* p = static_cast<const unsigned char*>(buf);
    size_t sent = 0;
    while (sent < n) {
        const ssize_t k = ::send(fd, p + sent, n - sent, MSG_NOSIGNAL);
        if (k < 0) {
            if (errno == EINTR) continue;
            fail("send");
        }
        sent += static_cast<size_t>(k);
    }
}

}  // namespace

Endpoint parse_endpoint(const std::string& url) {
    const std::string pre = "tcp://";
    const size_t colon = url.rfind(':');
    if (url.compare(0, pre.size(), pre) != 0 || colon == std::string::npos || colon <= pre.size())
        throw std::runtime_error("endpoint '" + url + "': expected tcp://host:port");
    Endpoint e;
    e.host = url.substr(pre.size(), colon - pre.size());
    const int port = std::stoi(url.substr(colon + 1));
    if (port < 0 || port > 65535) throw std::runtime_error("endpoint '" + url + "': bad port");
    e.port = static_cast<uint16_t>(port);
    return e;
}

uint64_t realtime_ns() {
    timespec t{};
    clock_gettime(CLOCK_REALTIME, &t);
    return static_cast<uint64_t>(t.tv_sec) * 1000000000ull + static_cast<uint64_t>(t.tv_nsec);
}

uint64_t monotonic_ns() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return static_cast<uint64_t>(t.tv_sec) * 1000000000ull + static_cast<uint64_t>(t.tv_nsec);
}

Socket& Socket::operator=(Socket&& o) noexcept {
    if (this != &o) {
        close();
        fd_ = o.fd_;
        o.fd_ = -1;
    }
    return *this;
}

Socket::~Socket() { close(); }

void Socket::close() {
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
}

bool Socket::read_frame(Frame& f) {
    if (!read_all(fd_, &f.h, sizeof f.h)) return false;
    if (f.h.magic != kFrameMagic || f.h.version != kFrameVersion2)
        throw std::runtime_error("bad frame header (magic or version): not a version 2 frame");
    if (f.h.dtype > static_cast<uint8_t>(DType::json)) throw std::runtime_error("bad frame header: dtype");
    f.payload.resize(payload_bytes(f.h));
    if (!f.payload.empty() && !read_all(fd_, f.payload.data(), f.payload.size()))
        throw std::runtime_error("connection closed inside a frame");
    return true;
}

void Socket::write_frame(const Frame& f) {
    if (f.payload.size() != payload_bytes(f.h)) throw std::logic_error("frame: payload size does not match its header");
    write_all(fd_, &f.h, sizeof f.h);
    if (!f.payload.empty()) write_all(fd_, f.payload.data(), f.payload.size());
}

bool Socket::wait_closed(double timeout_s) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_s);
    unsigned char buf[4096];
    for (;;) {
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
        if (left <= 0) return false;
        pollfd p{fd_, POLLIN, 0};
        const int r = ::poll(&p, 1, static_cast<int>(left));
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
        const ssize_t k = ::recv(fd_, buf, sizeof buf, 0);
        if (k == 0) return true;                        // the peer closed it
        if (k < 0 && errno != EINTR) return true;       // reset: closed as well
    }
}

bool Socket::read_line(std::string& line) {
    line.clear();
    for (;;) {
        char c;
        const ssize_t k = ::recv(fd_, &c, 1, 0);
        if (k == 0) return false;
        if (k < 0) {
            if (errno == EINTR) continue;
            fail("recv");
        }
        if (c == '\n') return true;
        line.push_back(c);
    }
}

void Socket::write_line(const std::string& line) {
    write_all(fd_, line.data(), line.size());
    write_all(fd_, "\n", 1);
}

Listener::Listener(const Endpoint& e) {
    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd_ < 0) fail("socket");
    const int one = 1;
    ::setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(e.port);
    if (e.host.empty() || e.host == "0.0.0.0" || e.host == "*") addr.sin_addr.s_addr = htonl(INADDR_ANY);
    else if (::inet_pton(AF_INET, e.host.c_str(), &addr.sin_addr) != 1)
        throw std::runtime_error("listen on '" + e.host + "': an IPv4 address or 0.0.0.0 is expected");
    if (::bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof addr) < 0) fail("bind to port " + std::to_string(e.port));
    if (::listen(fd_, 1) < 0) fail("listen");
    socklen_t len = sizeof addr;
    ::getsockname(fd_, reinterpret_cast<sockaddr*>(&addr), &len);
    port_ = ntohs(addr.sin_port);
}

Listener::~Listener() {
    if (fd_ >= 0) ::close(fd_);
}

Socket Listener::accept() {
    for (;;) {
        const int c = ::accept(fd_, nullptr, nullptr);
        if (c >= 0) {
            const int one = 1;
            ::setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
            return Socket(c);
        }
        if (errno != EINTR) fail("accept");
    }
}

Socket connect_retry(const Endpoint& e, double timeout_s) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_s);
    std::string last = "no attempt";
    for (;;) {
        addrinfo hints{}, *res = nullptr;
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        const int rc = ::getaddrinfo(e.host.c_str(), std::to_string(e.port).c_str(), &hints, &res);
        if (rc == 0) {
            for (addrinfo* a = res; a; a = a->ai_next) {
                const int fd = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
                if (fd < 0) continue;
                if (::connect(fd, a->ai_addr, a->ai_addrlen) == 0) {
                    ::freeaddrinfo(res);
                    const int one = 1;
                    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
                    return Socket(fd);
                }
                last = std::strerror(errno);
                ::close(fd);
            }
            ::freeaddrinfo(res);
        } else {
            last = ::gai_strerror(rc);
        }
        if (std::chrono::steady_clock::now() >= deadline)
            throw std::runtime_error("could not connect to tcp://" + e.host + ":" + std::to_string(e.port) + " within " +
                                     std::to_string(static_cast<int>(timeout_s)) + " s (" + last + ")");
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

}  // namespace arof
