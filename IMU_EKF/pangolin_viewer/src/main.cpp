#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
using SocketHandle = SOCKET;
using SocketLength = int;
constexpr SocketHandle kInvalidSocket = INVALID_SOCKET;
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
using SocketHandle = int;
using SocketLength = socklen_t;
constexpr SocketHandle kInvalidSocket = -1;
#endif

#include <pangolin/pangolin.h>

namespace {

constexpr const char* kProtocolHeader = "CV7GUI/1";
constexpr double kPi = 3.14159265358979323846;

struct Point2 {
    double east = 0.0;
    double north = 0.0;
};

struct Arguments {
    std::string bind_address = "0.0.0.0";
    int port = 5600;
    std::size_t history = 100000;
};

void close_socket(SocketHandle socket_handle) {
    if (socket_handle == kInvalidSocket) {
        return;
    }
#ifdef _WIN32
    closesocket(socket_handle);
#else
    close(socket_handle);
#endif
}

class SocketRuntime {
public:
    SocketRuntime() {
#ifdef _WIN32
        WSADATA data{};
        if (WSAStartup(MAKEWORD(2, 2), &data) != 0) {
            throw std::runtime_error("WSAStartup failed");
        }
#endif
    }

    ~SocketRuntime() {
#ifdef _WIN32
        WSACleanup();
#endif
    }

    SocketRuntime(const SocketRuntime&) = delete;
    SocketRuntime& operator=(const SocketRuntime&) = delete;
};

class UdpReceiver {
public:
    UdpReceiver(const std::string& bind_address, int port) : port_(port) {
        socket_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (socket_ == kInvalidSocket) {
            throw std::runtime_error("cannot create UDP socket");
        }

        int reuse = 1;
        setsockopt(socket_, SOL_SOCKET, SO_REUSEADDR,
                   reinterpret_cast<const char*>(&reuse), sizeof(reuse));

        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(static_cast<std::uint16_t>(port));
        if (inet_pton(AF_INET, bind_address.c_str(), &address.sin_addr) != 1) {
            close_socket(socket_);
            socket_ = kInvalidSocket;
            throw std::runtime_error("invalid --bind IPv4 address: " + bind_address);
        }
        if (bind(socket_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
            close_socket(socket_);
            socket_ = kInvalidSocket;
            throw std::runtime_error("cannot bind UDP port " + std::to_string(port));
        }

#ifdef _WIN32
        u_long nonblocking = 1;
        if (ioctlsocket(socket_, FIONBIO, &nonblocking) != 0) {
            throw std::runtime_error("cannot make UDP socket non-blocking");
        }
#else
        const int flags = fcntl(socket_, F_GETFL, 0);
        if (flags < 0 || fcntl(socket_, F_SETFL, flags | O_NONBLOCK) < 0) {
            throw std::runtime_error("cannot make UDP socket non-blocking");
        }
#endif
    }

    ~UdpReceiver() { close_socket(socket_); }

    UdpReceiver(const UdpReceiver&) = delete;
    UdpReceiver& operator=(const UdpReceiver&) = delete;

    std::optional<std::string> receive_latest() {
        std::optional<std::string> latest;
        std::array<char, 65536> buffer{};
        while (true) {
            sockaddr_in sender{};
            SocketLength sender_length = sizeof(sender);
#ifdef _WIN32
            const int received = recvfrom(
                socket_, buffer.data(), static_cast<int>(buffer.size() - 1), 0,
                reinterpret_cast<sockaddr*>(&sender), &sender_length);
            if (received == SOCKET_ERROR) {
                const int error = WSAGetLastError();
                if (error == WSAEWOULDBLOCK) {
                    break;
                }
                break;
            }
#else
            const ssize_t received = recvfrom(
                socket_, buffer.data(), buffer.size() - 1, 0,
                reinterpret_cast<sockaddr*>(&sender), &sender_length);
            if (received < 0) {
                break;
            }
#endif
            if (received <= 0) {
                break;
            }
            last_sender_ = sender;
            has_sender_ = true;
            latest = std::string(buffer.data(), static_cast<std::size_t>(received));
        }
        return latest;
    }

    bool send_stop_command() const {
        if (!has_sender_) {
            return false;
        }
        sockaddr_in target = last_sender_;
        target.sin_port = htons(static_cast<std::uint16_t>(port_ + 1));
        const std::string message = std::string(kProtocolHeader) + "\ncommand=stop\n";
#ifdef _WIN32
        const int sent = sendto(
            socket_, message.data(), static_cast<int>(message.size()), 0,
            reinterpret_cast<const sockaddr*>(&target), sizeof(target));
        return sent == static_cast<int>(message.size());
#else
        const ssize_t sent = sendto(
            socket_, message.data(), message.size(), 0,
            reinterpret_cast<const sockaddr*>(&target), sizeof(target));
        return sent == static_cast<ssize_t>(message.size());
#endif
    }

private:
    SocketHandle socket_ = kInvalidSocket;
    int port_ = 0;
    sockaddr_in last_sender_{};
    bool has_sender_ = false;
};

using Telemetry = std::map<std::string, std::string>;

std::optional<Telemetry> parse_telemetry(const std::string& payload) {
    std::istringstream stream(payload);
    std::string line;
    if (!std::getline(stream, line)) {
        return std::nullopt;
    }
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    if (line != kProtocolHeader) {
        return std::nullopt;
    }

    Telemetry values;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const std::size_t separator = line.find('=');
        if (separator == std::string::npos) {
            continue;
        }
        values[line.substr(0, separator)] = line.substr(separator + 1);
    }
    return values;
}

std::string text_value(const Telemetry& values, const std::string& key,
                       const std::string& fallback = "Waiting for data") {
    const auto found = values.find(key);
    return found == values.end() || found->second.empty() ? fallback : found->second;
}

std::optional<double> number_value(const Telemetry& values, const std::string& key) {
    const auto found = values.find(key);
    if (found == values.end() || found->second.empty()) {
        return std::nullopt;
    }
    char* end = nullptr;
    const double number = std::strtod(found->second.c_str(), &end);
    if (end == found->second.c_str() || *end != '\0' || !std::isfinite(number)) {
        return std::nullopt;
    }
    return number;
}

std::optional<std::uint64_t> integer_value(const Telemetry& values,
                                           const std::string& key) {
    const auto number = number_value(values, key);
    if (!number || *number < 0.0) {
        return std::nullopt;
    }
    return static_cast<std::uint64_t>(*number);
}

bool bool_value(const Telemetry& values, const std::string& key) {
    return text_value(values, key, "false") == "true";
}

std::string format_number(const Telemetry& values, const std::string& key,
                          int precision, const std::string& suffix = "") {
    const auto value = number_value(values, key);
    if (!value) {
        return "Waiting for data";
    }
    std::ostringstream text;
    text << std::fixed << std::setprecision(precision) << *value << suffix;
    return text.str();
}

std::string format_vector(const Telemetry& values,
                          const std::array<std::string, 3>& keys,
                          int precision, const std::string& suffix) {
    const auto x = number_value(values, keys[0]);
    const auto y = number_value(values, keys[1]);
    const auto z = number_value(values, keys[2]);
    if (!x || !y || !z) {
        return "Waiting for data";
    }
    std::ostringstream text;
    text << std::fixed << std::setprecision(precision) << *x << ", " << *y << ", "
         << *z << suffix;
    return text.str();
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        auto required_value = [&](const std::string& name) -> std::string {
            if (index + 1 >= argc) {
                throw std::runtime_error(name + " requires a value");
            }
            return argv[++index];
        };
        if (option == "--bind") {
            result.bind_address = required_value(option);
        } else if (option == "--port") {
            result.port = std::stoi(required_value(option));
        } else if (option == "--history") {
            result.history = static_cast<std::size_t>(
                std::stoull(required_value(option)));
        } else if (option == "--help" || option == "-h") {
            std::cout << "Usage: cv7_pangolin_viewer [--bind 0.0.0.0] "
                         "[--port 5600] [--history 100000]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + option);
        }
    }
    if (result.port < 1 || result.port > 65534) {
        throw std::runtime_error("--port must be between 1 and 65534");
    }
    if (result.history == 0) {
        throw std::runtime_error("--history must be positive");
    }
    return result;
}

template <typename T>
void limit_history(std::vector<T>& values, std::size_t maximum) {
    if (values.size() <= maximum) {
        return;
    }
    const std::size_t remove_count = values.size() - maximum;
    values.erase(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(remove_count));
}

void draw_grid(double extent = 100.0, double spacing = 5.0) {
    glLineWidth(1.0F);
    glColor3f(0.18F, 0.18F, 0.18F);
    glBegin(GL_LINES);
    for (double coordinate = -extent; coordinate <= extent; coordinate += spacing) {
        glVertex3d(coordinate, -extent, 0.0);
        glVertex3d(coordinate, extent, 0.0);
        glVertex3d(-extent, coordinate, 0.0);
        glVertex3d(extent, coordinate, 0.0);
    }
    glEnd();

    glLineWidth(2.0F);
    glBegin(GL_LINES);
    glColor3f(0.9F, 0.2F, 0.2F);
    glVertex3d(0.0, 0.0, 0.02);
    glVertex3d(5.0, 0.0, 0.02);
    glColor3f(0.2F, 0.9F, 0.2F);
    glVertex3d(0.0, 0.0, 0.02);
    glVertex3d(0.0, 5.0, 0.02);
    glEnd();
}

void draw_ekf_track(const std::vector<Point2>& track) {
    if (track.empty()) {
        return;
    }
    glColor3f(0.12F, 0.47F, 0.82F);
    glLineWidth(2.5F);
    glBegin(GL_LINE_STRIP);
    for (const Point2& point : track) {
        glVertex3d(point.east, point.north, 0.05);
    }
    glEnd();
}

void draw_gps_points(const std::vector<Point2>& points) {
    glColor3f(1.0F, 0.50F, 0.05F);
    glPointSize(5.0F);
    glBegin(GL_POINTS);
    for (const Point2& point : points) {
        glVertex3d(point.east, point.north, 0.08);
    }
    glEnd();
}

void draw_heading_arrow(const Point2& position, double yaw_degrees) {
    // Navigation yaw is clockwise from North. East = sin(yaw), North = cos(yaw).
    const double yaw = yaw_degrees * kPi / 180.0;
    const double east = std::sin(yaw);
    const double north = std::cos(yaw);
    const double side_east = std::sin(yaw + 2.55);
    const double side_north = std::cos(yaw + 2.55);
    const double other_east = std::sin(yaw - 2.55);
    const double other_north = std::cos(yaw - 2.55);

    glColor3f(0.2F, 0.9F, 0.95F);
    glLineWidth(3.0F);
    glBegin(GL_LINES);
    glVertex3d(position.east, position.north, 0.15);
    glVertex3d(position.east + 2.0 * east, position.north + 2.0 * north, 0.15);
    glVertex3d(position.east + 2.0 * east, position.north + 2.0 * north, 0.15);
    glVertex3d(position.east + 2.0 * east + 0.7 * side_east,
               position.north + 2.0 * north + 0.7 * side_north, 0.15);
    glVertex3d(position.east + 2.0 * east, position.north + 2.0 * north, 0.15);
    glVertex3d(position.east + 2.0 * east + 0.7 * other_east,
               position.north + 2.0 * north + 0.7 * other_north, 0.15);
    glEnd();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments arguments = parse_arguments(argc, argv);
        SocketRuntime socket_runtime;
        UdpReceiver receiver(arguments.bind_address, arguments.port);

        pangolin::CreateWindowAndBind("CV7 INS / mosaic-H Live Status", 1400, 850);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        constexpr int panel_width = 390;
        pangolin::CreatePanel("status")
            .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(panel_width));

        pangolin::Var<std::string> ui_recorder("status.Recorder", "Waiting for recorder");
        pangolin::Var<std::string> ui_cv7_connection("status.CV7 connection", "Waiting for data");
        pangolin::Var<std::string> ui_gps_connection("status.GPS connection", "Waiting for data");
        pangolin::Var<std::string> ui_imu_status("status.IMU stream", "Waiting for data");
        pangolin::Var<std::string> ui_ekf_status("status.EKF stream", "Waiting for data");
        pangolin::Var<std::string> ui_filter_state("status.Filter state", "Waiting for data");
        pangolin::Var<std::string> ui_filter_condition("status.Filter condition", "Waiting for data");
        pangolin::Var<std::string> ui_filter_warnings("status.Filter warnings", "Waiting for data");
        pangolin::Var<std::string> ui_aiding("status.Aiding summary", "Waiting for data");
        pangolin::Var<std::string> ui_position_aiding("status.Position aiding", "Waiting for data");
        pangolin::Var<std::string> ui_velocity_aiding("status.Velocity aiding", "Waiting for data");
        pangolin::Var<std::string> ui_heading("status.Heading", "Waiting for data");
        pangolin::Var<std::string> ui_gnss_fix("status.GNSS fix", "Waiting for data");
        pangolin::Var<std::string> ui_time_sync("status.Time synchronization", "Waiting for data");
        pangolin::Var<std::string> ui_rates("status.Data rates", "Waiting for data");
        pangolin::Var<std::string> ui_satellites("status.Satellites", "Waiting for data");
        pangolin::Var<std::string> ui_precision("status.Position precision", "Waiting for data");
        pangolin::Var<std::string> ui_speed("status.Speed", "Waiting for data");
        pangolin::Var<std::string> ui_yaw("status.INS yaw", "Waiting for data");
        pangolin::Var<std::string> ui_accel("status.Acceleration XYZ", "Waiting for data");
        pangolin::Var<std::string> ui_gyro("status.Angular rate XYZ", "Waiting for data");
        pangolin::Var<std::string> ui_rows("status.CSV rows", "Waiting for data");
        pangolin::Var<std::string> ui_legend("status.Track colors", "Blue: EKF   Orange: GPS raw");
        pangolin::Var<bool> ui_clear("status.Clear tracks", false, false);
        pangolin::Var<bool> ui_stop("status.Stop recording", false, false);
        pangolin::Var<bool> ui_quit("status.Quit viewer", false, false);

        pangolin::OpenGlRenderState camera(
            pangolin::ProjectionMatrix(1400 - panel_width, 850, 650.0, 650.0,
                                       (1400 - panel_width) / 2.0, 425.0, 0.1, 10000.0),
            pangolin::ModelViewLookAt(0.0, -35.0, -45.0, 0.0, 0.0, 0.0,
                                      pangolin::AxisNegY));
        pangolin::View& display = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(panel_width), 1.0,
                       -static_cast<float>(1400 - panel_width) / 850.0F)
            .SetHandler(new pangolin::Handler3D(camera));

        Telemetry telemetry;
        std::vector<Point2> ekf_track;
        std::vector<Point2> gps_points;
        std::uint64_t last_ekf_sample = 0;
        std::uint64_t last_gps_sample = 0;
        auto last_packet = std::chrono::steady_clock::time_point{};
        bool shutdown_received = false;

        while (!pangolin::ShouldQuit() && !static_cast<bool>(ui_quit)) {
            if (const auto payload = receiver.receive_latest()) {
                if (const auto parsed = parse_telemetry(*payload)) {
                    telemetry = *parsed;
                    last_packet = std::chrono::steady_clock::now();
                    shutdown_received = bool_value(telemetry, "shutdown");

                    const auto ekf_sample = integer_value(telemetry, "ekf_sample_id");
                    const auto ekf_east = number_value(telemetry, "ekf_east_m");
                    const auto ekf_north = number_value(telemetry, "ekf_north_m");
                    if (ekf_sample && *ekf_sample != last_ekf_sample && ekf_east && ekf_north) {
                        ekf_track.push_back({*ekf_east, *ekf_north});
                        last_ekf_sample = *ekf_sample;
                        limit_history(ekf_track, arguments.history);
                    }

                    const auto gps_sample = integer_value(telemetry, "gps_sample_id");
                    const auto gps_east = number_value(telemetry, "gps_east_m");
                    const auto gps_north = number_value(telemetry, "gps_north_m");
                    if (gps_sample && *gps_sample != last_gps_sample && gps_east && gps_north) {
                        gps_points.push_back({*gps_east, *gps_north});
                        last_gps_sample = *gps_sample;
                        limit_history(gps_points, arguments.history);
                    }
                }
            }

            const auto now = std::chrono::steady_clock::now();
            const bool connected = last_packet.time_since_epoch().count() != 0 &&
                now - last_packet < std::chrono::seconds(2);
            ui_recorder = connected ? text_value(telemetry, "recorder_status")
                                    : "Recorder disconnected";
            ui_cv7_connection = text_value(telemetry, "cv7_connection");
            ui_gps_connection = text_value(telemetry, "gps_connection");
            ui_imu_status = text_value(telemetry, "imu_status");
            ui_ekf_status = text_value(telemetry, "ekf_status");
            ui_filter_state = text_value(telemetry, "filter_state");
            ui_filter_condition = text_value(telemetry, "filter_condition");
            ui_filter_warnings = text_value(telemetry, "filter_warnings");
            ui_aiding = text_value(telemetry, "aiding_status");
            ui_position_aiding = text_value(telemetry, "position_aiding_status");
            ui_velocity_aiding = text_value(telemetry, "velocity_aiding_status");
            ui_heading = text_value(telemetry, "heading_status");
            ui_gnss_fix = text_value(telemetry, "gnss_fix");
            ui_time_sync = text_value(telemetry, "time_sync_status");

            ui_rates = "IMU " + format_number(telemetry, "cv7_imu_rate_hz", 1, " Hz") +
                       " | EKF " + format_number(telemetry, "cv7_ekf_rate_hz", 1, " Hz") +
                       " | NMEA " + format_number(telemetry, "gps_nmea_rate_hz", 1, " Hz");
            ui_satellites = format_number(telemetry, "satellites", 0) +
                            " | HDOP " + format_number(telemetry, "hdop", 2);
            ui_precision = "EKF " + format_number(telemetry, "position_uncertainty_m", 3, " m") +
                           " | GPS " + format_number(telemetry, "gst_horizontal_sigma_m", 3, " m");
            ui_speed = format_number(telemetry, "speed_mps", 3, " m/s");
            ui_yaw = format_number(telemetry, "yaw_deg", 2, " deg");
            ui_accel = format_vector(telemetry, {"accel_x", "accel_y", "accel_z"}, 4, " g");
            ui_gyro = format_vector(telemetry, {"gyro_x", "gyro_y", "gyro_z"}, 4, " rad/s");
            ui_rows = format_number(telemetry, "row_count", 0);

            if (pangolin::Pushed(ui_clear)) {
                ekf_track.clear();
                gps_points.clear();
            }
            if (pangolin::Pushed(ui_stop)) {
                ui_recorder = receiver.send_stop_command()
                    ? "Stop requested"
                    : "Cannot stop: no recorder address";
            }

            glClearColor(0.06F, 0.06F, 0.07F, 1.0F);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            display.Activate(camera);
            draw_grid();
            draw_ekf_track(ekf_track);
            draw_gps_points(gps_points);
            const auto yaw = number_value(telemetry, "yaw_deg");
            if (yaw && !ekf_track.empty()) {
                draw_heading_arrow(ekf_track.back(), *yaw);
            }
            pangolin::FinishFrame();

            if (shutdown_received) {
                break;
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cv7_pangolin_viewer: " << error.what() << '\n';
        return 1;
    }
}
