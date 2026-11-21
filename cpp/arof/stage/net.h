// TCP endpoints of the stages (PIPELINE.md, Section 3): "tcp://host:port" from the environment,
// a listener for a stage's input, a connection with retry for its output, and whole frames in and
// out of a socket.
#pragma once

#include <cstdint>
#include <string>

#include "stage/frame2.h"

namespace arof {

struct Endpoint {
    std::string host;
    uint16_t port = 0;
};
Endpoint parse_endpoint(const std::string& url);        // "tcp://host:port"

uint64_t realtime_ns();                                 // CLOCK_REALTIME
uint64_t monotonic_ns();                                // CLOCK_MONOTONIC

// a connected socket; closed by the destructor
class Socket {
public:
    Socket() = default;
    explicit Socket(int fd) : fd_(fd) {}
    Socket(Socket&& o) noexcept : fd_(o.fd_) { o.fd_ = -1; }
    Socket& operator=(Socket&& o) noexcept;
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;
    ~Socket();

    bool valid() const { return fd_ >= 0; }
    void close();
    // false: the peer closed the connection before the first byte of a frame; throws on a frame
    // cut short, a bad header or a socket error
    bool read_frame(Frame& f);
    void write_frame(const Frame& f);
    // after the END frame: waits until the peer closes the connection (true) or timeout_s passes (false)
    bool wait_closed(double timeout_s);
    // the control connections: one line of text each way
    bool read_line(std::string& line);
    void write_line(const std::string& line);

private:
    int fd_ = -1;
};

// listens on an endpoint (port 0: any free port, see port()); accept() takes one connection
class Listener {
public:
    explicit Listener(const Endpoint& e);
    ~Listener();
    uint16_t port() const { return port_; }
    Socket accept();

private:
    int fd_ = -1;
    uint16_t port_ = 0;
};

// connects, retrying until timeout_s has passed (compose does not order the start of the services)
Socket connect_retry(const Endpoint& e, double timeout_s);

}  // namespace arof
