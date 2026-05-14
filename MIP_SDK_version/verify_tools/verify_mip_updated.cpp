/*
 * MIP Stream Verifier for Teensy SBF to MIP bridge
 *
 * Purpose
 *   1. Read a binary capture created by record_mip.ps1.
 *   2. Use the official HBK MicroStrain MIP SDK parser and serializers.
 *   3. Verify that the Teensy USB mirror contains valid Aiding MIP packets.
 *   4. Verify that decoded fields match the rules used in main.cpp:
 *        Position LLH  : descriptor set 0x13, field 0x22
 *        Velocity NED  : descriptor set 0x13, field 0x29, Down is used instead of Up
 *        Heading True  : descriptor set 0x13, field 0x31, radians, optional
 *   5. Optionally compare decoded values against a GPS truth CSV.
 *
 * Important
 *   record_mip.ps1 records only the bytes coming from the Teensy USB port.
 *   In the current main.cpp, that stream is the generated MIP stream only.
 *   Therefore exact GPS comparison requires an additional truth CSV exported
 *   from the original GPS or SBF data.
 *
 * Build
 *   Use CMake and link this file against the HBK MicroStrain mip_sdk target mip.
 */

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "mip/mip_all.h"
#include "mip/definitions/commands_aiding.h"

using namespace mip::C;
using namespace microstrain::C;

namespace {

constexpr uint8_t EXPECTED_DESC_SET_AIDING = MIP_AIDING_CMD_DESC_SET;
constexpr double PI_D = 3.141592653589793238462643383279502884;
constexpr uint64_t NS_PER_MS = 1000000ULL;
constexpr uint64_t NS_PER_WEEK = 604800000000000ULL;

struct Options {
    std::string mip_path;
    std::string truth_csv_path;
    uint8_t expected_frame_id = 1;
    int expected_timebase = MIP_TIME_TIMEBASE_TIME_OF_ARRIVAL;
    bool check_timebase = true;
    bool strict_unknown_fields = true;
    bool strict_sequence = true;
    bool print_samples = true;
    size_t max_print_samples = 5;

    double latlon_tol_deg = 1e-8;
    double height_tol_m = 0.05;
    double vel_tol_mps = 0.02;
    double heading_tol_deg = 0.2;
};

struct Stats {
    size_t raw_bytes = 0;
    size_t scan_candidate_packets = 0;
    size_t scan_candidate_bytes = 0;
    size_t scan_noise_bytes = 0;
    size_t scan_truncated_packets = 0;

    uint32_t packet_total = 0;
    uint32_t packet_crc_error = 0;
    uint32_t packet_wrong_descriptor_set = 0;
    uint32_t packet_aiding = 0;

    uint32_t field_pos_llh = 0;
    uint32_t field_vel_ned = 0;
    uint32_t field_heading_true = 0;
    uint32_t field_frame_config = 0;
    uint32_t field_other = 0;

    uint32_t decode_error = 0;
    uint32_t range_error = 0;
    uint32_t frame_id_error = 0;
    uint32_t timebase_error = 0;
    uint32_t valid_flag_error = 0;
    uint32_t uncertainty_error = 0;
    uint32_t sequence_error = 0;
    uint32_t truth_mismatch = 0;
};

struct TimeInfo {
    int timebase = 0;
    uint8_t reserved = 0;
    uint64_t nanoseconds = 0;
};

struct PosSample {
    TimeInfo time;
    uint8_t frame_id = 0;
    double lat_deg = 0;
    double lon_deg = 0;
    double height_m = 0;
    float sigma_n = 0;
    float sigma_e = 0;
    float sigma_d = 0;
    uint16_t valid_flags = 0;
};

struct VelSample {
    TimeInfo time;
    uint8_t frame_id = 0;
    float vn = 0;
    float ve = 0;
    float vd = 0;
    float sigma_vn = 0;
    float sigma_ve = 0;
    float sigma_vd = 0;
    uint16_t valid_flags = 0;
};

struct HeadingSample {
    TimeInfo time;
    uint8_t frame_id = 0;
    float heading_rad = 0;
    float sigma_rad = 0;
    uint16_t valid_flags = 0;
};

enum class EventKind { Pos, Vel, Heading };

struct Event {
    EventKind kind;
    uint32_t packet_number = 0;
    PosSample pos;
    VelSample vel;
    HeadingSample heading;
};

struct Epoch {
    bool has_pos = false;
    bool has_vel = false;
    bool has_heading = false;
    PosSample pos;
    VelSample vel;
    HeadingSample heading;
};

struct TruthRow {
    bool has_lat = false;
    bool has_lon = false;
    bool has_height = false;
    bool has_vn = false;
    bool has_ve = false;
    bool has_vu = false;
    bool has_vd = false;
    bool has_heading_deg = false;
    bool has_week = false;
    bool has_tow_ms = false;

    double lat_deg = 0;
    double lon_deg = 0;
    double height_m = 0;
    double vn = 0;
    double ve = 0;
    double vu = 0;
    double vd = 0;
    double heading_deg = 0;
    uint16_t week = 0;
    uint32_t tow_ms = 0;
};

Options g_options;
Stats g_stats;
std::vector<Event> g_events;

std::string lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

std::string trim(const std::string& s) {
    size_t first = 0;
    while (first < s.size() && std::isspace(static_cast<unsigned char>(s[first]))) {
        ++first;
    }
    size_t last = s.size();
    while (last > first && std::isspace(static_cast<unsigned char>(s[last - 1]))) {
        --last;
    }
    return s.substr(first, last - first);
}

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> out;
    std::string field;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                field.push_back('"');
                ++i;
            } else {
                in_quotes = !in_quotes;
            }
        } else if (c == ',' && !in_quotes) {
            out.push_back(trim(field));
            field.clear();
        } else {
            field.push_back(c);
        }
    }
    out.push_back(trim(field));
    return out;
}

bool parse_double(const std::string& s, double& value) {
    char* end = nullptr;
    value = std::strtod(s.c_str(), &end);
    return end != s.c_str() && *end == '\0' && std::isfinite(value);
}

bool parse_uint32(const std::string& s, uint32_t& value) {
    char* end = nullptr;
    unsigned long v = std::strtoul(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0') {
        return false;
    }
    value = static_cast<uint32_t>(v);
    return true;
}

bool parse_uint16(const std::string& s, uint16_t& value) {
    uint32_t v = 0;
    if (!parse_uint32(s, v) || v > 65535U) {
        return false;
    }
    value = static_cast<uint16_t>(v);
    return true;
}

bool approx_equal(double a, double b, double tol) {
    return std::fabs(a - b) <= tol;
}

double rad_to_deg(double rad) {
    return rad * 180.0 / PI_D;
}

double wrap_angle_deg(double x) {
    while (x > 180.0) x -= 360.0;
    while (x < -180.0) x += 360.0;
    return x;
}

uint64_t gps_time_ns(uint16_t week, uint32_t tow_ms) {
    return static_cast<uint64_t>(week) * NS_PER_WEEK + static_cast<uint64_t>(tow_ms) * NS_PER_MS;
}

TimeInfo to_time_info(const mip_time& t) {
    TimeInfo out;
    out.timebase = static_cast<int>(t.timebase);
    out.reserved = t.reserved;
    out.nanoseconds = t.nanoseconds;
    return out;
}

bool same_time(const TimeInfo& a, const TimeInfo& b) {
    return a.timebase == b.timebase && a.reserved == b.reserved && a.nanoseconds == b.nanoseconds;
}

void add_range_error(const std::string& msg) {
    ++g_stats.range_error;
    std::cerr << "[RANGE] " << msg << "\n";
}

void add_decode_error(const std::string& msg) {
    ++g_stats.decode_error;
    std::cerr << "[DECODE] " << msg << "\n";
}

void add_sequence_error(const std::string& msg) {
    ++g_stats.sequence_error;
    std::cerr << "[SEQUENCE] " << msg << "\n";
}

void check_common_time_and_frame(const TimeInfo& time, uint8_t frame_id, const char* label) {
    if (frame_id != g_options.expected_frame_id) {
        ++g_stats.frame_id_error;
        std::cerr << "[FRAME] " << label << " frame_id=" << static_cast<int>(frame_id)
                  << " expected=" << static_cast<int>(g_options.expected_frame_id) << "\n";
    }
    if (time.reserved != 1U) {
        ++g_stats.timebase_error;
        std::cerr << "[TIME] " << label << " reserved=" << static_cast<int>(time.reserved)
                  << " expected=1\n";
    }
    if (g_options.check_timebase && time.timebase != g_options.expected_timebase) {
        ++g_stats.timebase_error;
        std::cerr << "[TIME] " << label << " timebase=" << time.timebase
                  << " expected=" << g_options.expected_timebase << "\n";
    }
}

void validate_pos(const PosSample& s) {
    check_common_time_and_frame(s.time, s.frame_id, "Position LLH");
    if (s.valid_flags != MIP_AIDING_POS_LLH_COMMAND_VALID_FLAGS_ALL) {
        ++g_stats.valid_flag_error;
        std::cerr << "[FLAGS] Position LLH valid_flags=0x" << std::hex << s.valid_flags
                  << std::dec << " expected=0x" << std::hex
                  << MIP_AIDING_POS_LLH_COMMAND_VALID_FLAGS_ALL << std::dec << "\n";
    }
    if (!std::isfinite(s.lat_deg) || s.lat_deg < -90.0 || s.lat_deg > 90.0) {
        add_range_error("Latitude out of range");
    }
    if (!std::isfinite(s.lon_deg) || s.lon_deg < -180.0 || s.lon_deg > 180.0) {
        add_range_error("Longitude out of range");
    }
    if (!std::isfinite(s.height_m) || s.height_m < -1000.0 || s.height_m > 100000.0) {
        add_range_error("Height out of reasonable range");
    }
    if (!(std::isfinite(s.sigma_n) && std::isfinite(s.sigma_e) && std::isfinite(s.sigma_d)) ||
        s.sigma_n <= 0.0f || s.sigma_e <= 0.0f || s.sigma_d <= 0.0f) {
        ++g_stats.uncertainty_error;
        std::cerr << "[SIGMA] Position uncertainty must be positive when valid_flags are ALL\n";
    }
}

void validate_vel(const VelSample& s) {
    check_common_time_and_frame(s.time, s.frame_id, "Velocity NED");
    if (s.valid_flags != MIP_AIDING_VEL_NED_COMMAND_VALID_FLAGS_ALL) {
        ++g_stats.valid_flag_error;
        std::cerr << "[FLAGS] Velocity NED valid_flags=0x" << std::hex << s.valid_flags
                  << std::dec << " expected=0x" << std::hex
                  << MIP_AIDING_VEL_NED_COMMAND_VALID_FLAGS_ALL << std::dec << "\n";
    }
    if (!std::isfinite(s.vn) || !std::isfinite(s.ve) || !std::isfinite(s.vd)) {
        add_range_error("Velocity contains non finite value");
    }
    if (std::fabs(s.vn) > 500.0f || std::fabs(s.ve) > 500.0f || std::fabs(s.vd) > 500.0f) {
        add_range_error("Velocity is outside a conservative vehicle range of 500 m/s");
    }
    if (!(std::isfinite(s.sigma_vn) && std::isfinite(s.sigma_ve) && std::isfinite(s.sigma_vd)) ||
        s.sigma_vn <= 0.0f || s.sigma_ve <= 0.0f || s.sigma_vd <= 0.0f) {
        ++g_stats.uncertainty_error;
        std::cerr << "[SIGMA] Velocity uncertainty must be positive when valid_flags are ALL\n";
    }
}

void validate_heading(const HeadingSample& s) {
    check_common_time_and_frame(s.time, s.frame_id, "Heading True");
    if (s.valid_flags != 0x0001U) {
        ++g_stats.valid_flag_error;
        std::cerr << "[FLAGS] Heading valid_flags=0x" << std::hex << s.valid_flags
                  << std::dec << " expected=0x0001\n";
    }
    if (!std::isfinite(s.heading_rad) || s.heading_rad < -PI_D || s.heading_rad > PI_D) {
        add_range_error("Heading is outside the valid radian range");
    }
    if (!std::isfinite(s.sigma_rad) || s.sigma_rad <= 0.0f) {
        ++g_stats.uncertainty_error;
        std::cerr << "[SIGMA] Heading uncertainty must be positive when valid flag is set\n";
    }
}

bool extract_pos(const uint8_t* payload, uint8_t payload_len, PosSample& out) {
    mip_aiding_pos_llh_command cmd;
    std::memset(&cmd, 0, sizeof(cmd));
    mip_serializer s;
    microstrain_serializer_init_extraction(&s, payload, payload_len);
    extract_mip_aiding_pos_llh_command(&s, &cmd);
    if (!microstrain_serializer_is_ok(&s) || !microstrain_serializer_is_complete(&s)) {
        return false;
    }
    out.time = to_time_info(cmd.time);
    out.frame_id = cmd.frame_id;
    out.lat_deg = cmd.latitude;
    out.lon_deg = cmd.longitude;
    out.height_m = cmd.height;
    out.sigma_n = cmd.uncertainty[0];
    out.sigma_e = cmd.uncertainty[1];
    out.sigma_d = cmd.uncertainty[2];
    out.valid_flags = cmd.valid_flags;
    return true;
}

bool extract_vel(const uint8_t* payload, uint8_t payload_len, VelSample& out) {
    mip_aiding_vel_ned_command cmd;
    std::memset(&cmd, 0, sizeof(cmd));
    mip_serializer s;
    microstrain_serializer_init_extraction(&s, payload, payload_len);
    extract_mip_aiding_vel_ned_command(&s, &cmd);
    if (!microstrain_serializer_is_ok(&s) || !microstrain_serializer_is_complete(&s)) {
        return false;
    }
    out.time = to_time_info(cmd.time);
    out.frame_id = cmd.frame_id;
    out.vn = cmd.velocity[0];
    out.ve = cmd.velocity[1];
    out.vd = cmd.velocity[2];
    out.sigma_vn = cmd.uncertainty[0];
    out.sigma_ve = cmd.uncertainty[1];
    out.sigma_vd = cmd.uncertainty[2];
    out.valid_flags = cmd.valid_flags;
    return true;
}

bool extract_heading(const uint8_t* payload, uint8_t payload_len, HeadingSample& out) {
    mip_aiding_heading_true_command cmd;
    std::memset(&cmd, 0, sizeof(cmd));
    mip_serializer s;
    microstrain_serializer_init_extraction(&s, payload, payload_len);
    extract_mip_aiding_heading_true_command(&s, &cmd);
    if (!microstrain_serializer_is_ok(&s) || !microstrain_serializer_is_complete(&s)) {
        return false;
    }
    out.time = to_time_info(cmd.time);
    out.frame_id = cmd.frame_id;
    out.heading_rad = cmd.heading;
    out.sigma_rad = cmd.uncertainty;
    out.valid_flags = cmd.valid_flags;
    return true;
}

void process_aiding_field(const mip_field_view* field) {
    const uint8_t desc = mip_field_field_descriptor(field);
    const uint8_t* payload = mip_field_payload(field);
    const uint8_t payload_len = mip_field_payload_length(field);

    switch (desc) {
        case MIP_CMD_DESC_AIDING_POS_LLH: {
            ++g_stats.field_pos_llh;
            PosSample pos;
            if (!extract_pos(payload, payload_len, pos)) {
                add_decode_error("Position LLH payload could not be decoded completely");
                break;
            }
            validate_pos(pos);
            Event e;
            e.kind = EventKind::Pos;
            e.packet_number = g_stats.packet_total;
            e.pos = pos;
            g_events.push_back(e);
            break;
        }
        case MIP_CMD_DESC_AIDING_VEL_NED: {
            ++g_stats.field_vel_ned;
            VelSample vel;
            if (!extract_vel(payload, payload_len, vel)) {
                add_decode_error("Velocity NED payload could not be decoded completely");
                break;
            }
            validate_vel(vel);
            Event e;
            e.kind = EventKind::Vel;
            e.packet_number = g_stats.packet_total;
            e.vel = vel;
            g_events.push_back(e);
            break;
        }
        case MIP_CMD_DESC_AIDING_HEADING_TRUE: {
            ++g_stats.field_heading_true;
            HeadingSample heading;
            if (!extract_heading(payload, payload_len, heading)) {
                add_decode_error("Heading True payload could not be decoded completely");
                break;
            }
            validate_heading(heading);
            Event e;
            e.kind = EventKind::Heading;
            e.packet_number = g_stats.packet_total;
            e.heading = heading;
            g_events.push_back(e);
            break;
        }
        case MIP_CMD_DESC_AIDING_FRAME_CONFIG:
            ++g_stats.field_frame_config;
            if (g_options.strict_unknown_fields) {
                ++g_stats.field_other;
                std::cerr << "[FIELD] Frame Config was found in capture. Current main.cpp does not mirror setup commands to USB.\n";
            }
            break;
        default:
            ++g_stats.field_other;
            std::cerr << "[FIELD] Unexpected Aiding field descriptor 0x"
                      << std::hex << static_cast<int>(desc) << std::dec << "\n";
            break;
    }
}

bool packet_callback(void*, const mip_packet_view* packet, mip_timestamp) {
    ++g_stats.packet_total;

    if (!mip_packet_is_valid(packet)) {
        ++g_stats.packet_crc_error;
        return true;
    }

    const uint8_t desc_set = mip_packet_descriptor_set(packet);
    if (desc_set != EXPECTED_DESC_SET_AIDING) {
        ++g_stats.packet_wrong_descriptor_set;
        std::cerr << "[PACKET] Unexpected descriptor set 0x"
                  << std::hex << static_cast<int>(desc_set) << std::dec
                  << ". Expected Aiding 0x13.\n";
        return true;
    }

    ++g_stats.packet_aiding;

    mip_field_view field = mip_field_first_from_packet(packet);
    while (mip_field_is_valid(&field)) {
        process_aiding_field(&field);
        mip_field_next(&field);
    }
    return true;
}

void scan_raw_mip_candidates(const std::vector<uint8_t>& data) {
    g_stats.raw_bytes = data.size();
    size_t i = 0;
    while (i < data.size()) {
        if (i + 3 < data.size() && data[i] == 0x75 && data[i + 1] == 0x65) {
            const size_t payload_len = data[i + 3];
            const size_t packet_len = 4 + payload_len + 2;
            if (i + packet_len <= data.size()) {
                ++g_stats.scan_candidate_packets;
                g_stats.scan_candidate_bytes += packet_len;
                i += packet_len;
            } else {
                ++g_stats.scan_truncated_packets;
                g_stats.scan_noise_bytes += data.size() - i;
                break;
            }
        } else {
            ++g_stats.scan_noise_bytes;
            ++i;
        }
    }
}

std::vector<Epoch> build_epochs_from_events() {
    std::vector<Epoch> epochs;
    Epoch current;
    bool active = false;

    auto finish_current = [&]() {
        if (active) {
            if (!current.has_pos || !current.has_vel) {
                add_sequence_error("Incomplete epoch. Expected Position followed by Velocity.");
            }
            epochs.push_back(current);
            current = Epoch{};
            active = false;
        }
    };

    for (const Event& e : g_events) {
        if (e.kind == EventKind::Pos) {
            finish_current();
            current = Epoch{};
            current.has_pos = true;
            current.pos = e.pos;
            active = true;
        } else if (e.kind == EventKind::Vel) {
            if (!active || !current.has_pos || current.has_vel) {
                add_sequence_error("Velocity appeared without a matching preceding Position.");
                current = Epoch{};
                active = true;
            }
            current.has_vel = true;
            current.vel = e.vel;
        } else if (e.kind == EventKind::Heading) {
            if (!active || !current.has_pos || !current.has_vel || current.has_heading) {
                add_sequence_error("Heading appeared without a complete Position and Velocity pair.");
                continue;
            }
            current.has_heading = true;
            current.heading = e.heading;
        }
    }
    finish_current();

    for (size_t i = 0; i < epochs.size(); ++i) {
        const Epoch& ep = epochs[i];
        if (ep.has_pos && ep.has_vel && !same_time(ep.pos.time, ep.vel.time)) {
            add_sequence_error("Position and Velocity time fields differ within epoch " + std::to_string(i));
        }
        if (ep.has_pos && ep.has_heading && !same_time(ep.pos.time, ep.heading.time)) {
            add_sequence_error("Position and Heading time fields differ within epoch " + std::to_string(i));
        }
    }

    return epochs;
}

std::map<std::string, size_t> build_header_map(const std::vector<std::string>& header) {
    std::map<std::string, size_t> out;
    for (size_t i = 0; i < header.size(); ++i) {
        std::string key = lower_copy(trim(header[i]));
        out[key] = i;
    }
    return out;
}

bool get_column(const std::vector<std::string>& row,
                const std::map<std::string, size_t>& header,
                const std::vector<std::string>& names,
                std::string& value) {
    for (const std::string& name : names) {
        auto it = header.find(name);
        if (it != header.end() && it->second < row.size()) {
            value = trim(row[it->second]);
            if (!value.empty()) {
                return true;
            }
        }
    }
    return false;
}

std::vector<TruthRow> load_truth_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("Cannot open truth CSV: " + path);
    }

    std::string line;
    std::vector<std::string> header;
    while (std::getline(f, line)) {
        line = trim(line);
        if (!line.empty() && line[0] != '#') {
            header = split_csv_line(line);
            break;
        }
    }
    if (header.empty()) {
        throw std::runtime_error("Truth CSV has no header");
    }

    auto map = build_header_map(header);
    std::vector<TruthRow> truth;

    while (std::getline(f, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::vector<std::string> row = split_csv_line(line);
        TruthRow t;
        std::string v;
        double d = 0;
        uint16_t u16 = 0;
        uint32_t u32 = 0;

        if (get_column(row, map, {"lat_deg", "latitude_deg", "latitude", "lat"}, v) && parse_double(v, d)) {
            t.has_lat = true;
            t.lat_deg = d;
        }
        if (get_column(row, map, {"lon_deg", "longitude_deg", "longitude", "lon"}, v) && parse_double(v, d)) {
            t.has_lon = true;
            t.lon_deg = d;
        }
        if (get_column(row, map, {"height_m", "h_m", "height", "ellipsoid_height_m"}, v) && parse_double(v, d)) {
            t.has_height = true;
            t.height_m = d;
        }
        if (get_column(row, map, {"vn", "vel_n", "velocity_n", "north_mps"}, v) && parse_double(v, d)) {
            t.has_vn = true;
            t.vn = d;
        }
        if (get_column(row, map, {"ve", "vel_e", "velocity_e", "east_mps"}, v) && parse_double(v, d)) {
            t.has_ve = true;
            t.ve = d;
        }
        if (get_column(row, map, {"vu", "vel_u", "velocity_u", "up_mps"}, v) && parse_double(v, d)) {
            t.has_vu = true;
            t.vu = d;
        }
        if (get_column(row, map, {"vd", "vel_d", "velocity_d", "down_mps"}, v) && parse_double(v, d)) {
            t.has_vd = true;
            t.vd = d;
        }
        if (get_column(row, map, {"heading_deg", "heading", "true_heading_deg"}, v) && parse_double(v, d)) {
            t.has_heading_deg = true;
            t.heading_deg = d;
        }
        if (get_column(row, map, {"week", "gps_week"}, v) && parse_uint16(v, u16)) {
            t.has_week = true;
            t.week = u16;
        }
        if (get_column(row, map, {"tow_ms", "gps_tow_ms"}, v) && parse_uint32(v, u32)) {
            t.has_tow_ms = true;
            t.tow_ms = u32;
        }

        truth.push_back(t);
    }

    return truth;
}

void report_truth_mismatch(size_t i, const std::string& msg) {
    ++g_stats.truth_mismatch;
    std::cerr << "[TRUTH] row=" << i << " " << msg << "\n";
}

void compare_against_truth(const std::vector<Epoch>& epochs, const std::vector<TruthRow>& truth) {
    const size_t n = std::min(epochs.size(), truth.size());
    if (epochs.size() != truth.size()) {
        report_truth_mismatch(n, "count mismatch: decoded epochs=" + std::to_string(epochs.size()) +
                                 " truth rows=" + std::to_string(truth.size()));
    }

    for (size_t i = 0; i < n; ++i) {
        const Epoch& ep = epochs[i];
        const TruthRow& t = truth[i];
        if (!ep.has_pos || !ep.has_vel) {
            report_truth_mismatch(i, "epoch is incomplete");
            continue;
        }
        if (t.has_lat && !approx_equal(ep.pos.lat_deg, t.lat_deg, g_options.latlon_tol_deg)) {
            report_truth_mismatch(i, "latitude mismatch mip=" + std::to_string(ep.pos.lat_deg) +
                                     " truth=" + std::to_string(t.lat_deg));
        }
        if (t.has_lon && !approx_equal(ep.pos.lon_deg, t.lon_deg, g_options.latlon_tol_deg)) {
            report_truth_mismatch(i, "longitude mismatch mip=" + std::to_string(ep.pos.lon_deg) +
                                     " truth=" + std::to_string(t.lon_deg));
        }
        if (t.has_height && !approx_equal(ep.pos.height_m, t.height_m, g_options.height_tol_m)) {
            report_truth_mismatch(i, "height mismatch mip=" + std::to_string(ep.pos.height_m) +
                                     " truth=" + std::to_string(t.height_m));
        }
        if (t.has_vn && !approx_equal(ep.vel.vn, t.vn, g_options.vel_tol_mps)) {
            report_truth_mismatch(i, "north velocity mismatch mip=" + std::to_string(ep.vel.vn) +
                                     " truth=" + std::to_string(t.vn));
        }
        if (t.has_ve && !approx_equal(ep.vel.ve, t.ve, g_options.vel_tol_mps)) {
            report_truth_mismatch(i, "east velocity mismatch mip=" + std::to_string(ep.vel.ve) +
                                     " truth=" + std::to_string(t.ve));
        }
        if (t.has_vu) {
            const double expected_vd = -t.vu;
            if (!approx_equal(ep.vel.vd, expected_vd, g_options.vel_tol_mps)) {
                report_truth_mismatch(i, "down velocity mismatch. main.cpp expects vd = negative vu. mip=" +
                                         std::to_string(ep.vel.vd) + " expected=" + std::to_string(expected_vd));
            }
        }
        if (t.has_vd && !approx_equal(ep.vel.vd, t.vd, g_options.vel_tol_mps)) {
            report_truth_mismatch(i, "down velocity mismatch mip=" + std::to_string(ep.vel.vd) +
                                     " truth=" + std::to_string(t.vd));
        }
        if (t.has_heading_deg) {
            if (!ep.has_heading) {
                report_truth_mismatch(i, "truth has heading but MIP epoch has no Heading True field");
            } else {
                const double mip_heading_deg = rad_to_deg(ep.heading.heading_rad);
                const double diff = wrap_angle_deg(mip_heading_deg - t.heading_deg);
                if (std::fabs(diff) > g_options.heading_tol_deg) {
                    report_truth_mismatch(i, "heading mismatch mip_deg=" + std::to_string(mip_heading_deg) +
                                             " truth_deg=" + std::to_string(t.heading_deg));
                }
            }
        }
        if (t.has_week && t.has_tow_ms && ep.pos.time.nanoseconds != 0) {
            const uint64_t expected_ns = gps_time_ns(t.week, t.tow_ms);
            if (ep.pos.time.nanoseconds != expected_ns) {
                report_truth_mismatch(i, "GPS time nanoseconds mismatch mip=" +
                                         std::to_string(ep.pos.time.nanoseconds) +
                                         " expected=" + std::to_string(expected_ns));
            }
        }
    }
}

void print_sample_epochs(const std::vector<Epoch>& epochs) {
    const size_t n = std::min(epochs.size(), g_options.max_print_samples);
    if (n == 0 || !g_options.print_samples) {
        return;
    }

    std::cout << "\nFirst decoded epochs\n";
    std::cout << std::fixed << std::setprecision(8);
    for (size_t i = 0; i < n; ++i) {
        const Epoch& ep = epochs[i];
        std::cout << "  [" << i << "] ";
        if (ep.has_pos) {
            std::cout << "lat=" << ep.pos.lat_deg
                      << " lon=" << ep.pos.lon_deg
                      << " h=" << std::setprecision(3) << ep.pos.height_m
                      << " sig_pos=" << ep.pos.sigma_n << "/" << ep.pos.sigma_e << "/" << ep.pos.sigma_d
                      << std::setprecision(8);
        }
        if (ep.has_vel) {
            std::cout << " vn=" << std::setprecision(3) << ep.vel.vn
                      << " ve=" << ep.vel.ve
                      << " vd=" << ep.vel.vd
                      << " sig_vel=" << ep.vel.sigma_vn << "/" << ep.vel.sigma_ve << "/" << ep.vel.sigma_vd
                      << std::setprecision(8);
        }
        if (ep.has_heading) {
            std::cout << " heading_deg=" << std::setprecision(3) << rad_to_deg(ep.heading.heading_rad)
                      << " sig_heading_deg=" << rad_to_deg(ep.heading.sigma_rad)
                      << std::setprecision(8);
        }
        std::cout << " timebase=" << (ep.has_pos ? ep.pos.time.timebase : 0)
                  << " ns=" << (ep.has_pos ? ep.pos.time.nanoseconds : 0)
                  << "\n";
    }
}

void print_usage() {
    std::cout <<
        "Usage:\n"
        "  verify_mip <mip_capture.bin> [options]\n\n"
        "Options:\n"
        "  --truth <gps_truth.csv>       Compare decoded MIP values with GPS truth rows by order\n"
        "  --frame-id <id>               Expected aiding frame ID. Default 1\n"
        "  --timebase <id>               Expected MIP timebase. Default 3 for TIME_OF_ARRIVAL\n"
        "  --no-timebase-check           Do not enforce timebase value\n"
        "  --latlon-tol-deg <value>      Default 1e-8\n"
        "  --height-tol-m <value>        Default 0.05\n"
        "  --vel-tol-mps <value>         Default 0.02\n"
        "  --heading-tol-deg <value>     Default 0.2\n"
        "  --max-samples <n>             Number of decoded epochs to print. Default 5\n\n"
        "Truth CSV columns accepted:\n"
        "  lat_deg, lon_deg, height_m, vn, ve, vu or vd, heading_deg, week, tow_ms\n";
}

bool parse_options(int argc, char** argv, Options& opt) {
    if (argc < 2) {
        print_usage();
        return false;
    }
    opt.mip_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value after ") + name);
            }
            return argv[++i];
        };
        if (a == "--truth") {
            opt.truth_csv_path = need_value("--truth");
        } else if (a == "--frame-id") {
            int v = std::atoi(need_value("--frame-id"));
            if (v < 0 || v > 255) throw std::runtime_error("frame id must be 0 to 255");
            opt.expected_frame_id = static_cast<uint8_t>(v);
        } else if (a == "--timebase") {
            opt.expected_timebase = std::atoi(need_value("--timebase"));
            opt.check_timebase = true;
        } else if (a == "--no-timebase-check") {
            opt.check_timebase = false;
        } else if (a == "--latlon-tol-deg") {
            opt.latlon_tol_deg = std::atof(need_value("--latlon-tol-deg"));
        } else if (a == "--height-tol-m") {
            opt.height_tol_m = std::atof(need_value("--height-tol-m"));
        } else if (a == "--vel-tol-mps") {
            opt.vel_tol_mps = std::atof(need_value("--vel-tol-mps"));
        } else if (a == "--heading-tol-deg") {
            opt.heading_tol_deg = std::atof(need_value("--heading-tol-deg"));
        } else if (a == "--max-samples") {
            opt.max_print_samples = static_cast<size_t>(std::strtoul(need_value("--max-samples"), nullptr, 10));
        } else if (a == "--help" || a == "/?") {
            print_usage();
            return false;
        } else {
            throw std::runtime_error("Unknown option: " + a);
        }
    }
    return true;
}

std::vector<uint8_t> read_binary_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

void print_summary(const std::vector<Epoch>& epochs, bool truth_used) {
    std::cout << "\n==============================================\n";
    std::cout << "SUMMARY\n";
    std::cout << "==============================================\n";
    std::cout << "Bytes in file                 : " << g_stats.raw_bytes << "\n";
    std::cout << "Raw MIP candidates            : " << g_stats.scan_candidate_packets << "\n";
    std::cout << "Raw non MIP or skipped bytes  : " << g_stats.scan_noise_bytes << "\n";
    std::cout << "Raw truncated candidates      : " << g_stats.scan_truncated_packets << "\n";
    std::cout << "SDK parser packets            : " << g_stats.packet_total << "\n";
    std::cout << "CRC errors                    : " << g_stats.packet_crc_error << "\n";
    std::cout << "Wrong descriptor set packets  : " << g_stats.packet_wrong_descriptor_set << "\n";
    std::cout << "Aiding packets                : " << g_stats.packet_aiding << "\n";
    std::cout << "Position LLH fields           : " << g_stats.field_pos_llh << "\n";
    std::cout << "Velocity NED fields           : " << g_stats.field_vel_ned << "\n";
    std::cout << "Heading True fields           : " << g_stats.field_heading_true << "\n";
    std::cout << "Frame Config fields           : " << g_stats.field_frame_config << "\n";
    std::cout << "Other fields                  : " << g_stats.field_other << "\n";
    std::cout << "Decoded epochs                : " << epochs.size() << "\n";
    std::cout << "Decode errors                 : " << g_stats.decode_error << "\n";
    std::cout << "Range errors                  : " << g_stats.range_error << "\n";
    std::cout << "Frame ID errors               : " << g_stats.frame_id_error << "\n";
    std::cout << "Timebase errors               : " << g_stats.timebase_error << "\n";
    std::cout << "Valid flag errors             : " << g_stats.valid_flag_error << "\n";
    std::cout << "Uncertainty errors            : " << g_stats.uncertainty_error << "\n";
    std::cout << "Sequence errors               : " << g_stats.sequence_error << "\n";
    std::cout << "Truth mismatches              : " << g_stats.truth_mismatch << "\n";
    std::cout << "Truth CSV used                : " << (truth_used ? "yes" : "no") << "\n";
    std::cout << "==============================================\n";
}

bool overall_ok(bool truth_used) {
    if (g_stats.raw_bytes == 0) return false;
    if (g_stats.packet_total == 0) return false;
    if (g_stats.packet_crc_error != 0) return false;
    if (g_stats.packet_wrong_descriptor_set != 0) return false;
    if (g_stats.field_other != 0) return false;
    if (g_stats.field_pos_llh == 0 || g_stats.field_vel_ned == 0) return false;
    if (g_stats.field_pos_llh != g_stats.field_vel_ned) return false;
    if (g_stats.decode_error != 0) return false;
    if (g_stats.range_error != 0) return false;
    if (g_stats.frame_id_error != 0) return false;
    if (g_stats.timebase_error != 0) return false;
    if (g_stats.valid_flag_error != 0) return false;
    if (g_stats.uncertainty_error != 0) return false;
    if (g_stats.sequence_error != 0) return false;
    if (truth_used && g_stats.truth_mismatch != 0) return false;
    if (g_stats.scan_truncated_packets != 0) return false;
    return true;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (!parse_options(argc, argv, g_options)) {
            return 1;
        }

        std::vector<uint8_t> data = read_binary_file(g_options.mip_path);
        if (data.empty()) {
            std::cerr << "[ERROR] File is empty\n";
            return 1;
        }

        std::cout << "==============================================\n";
        std::cout << "MIP Stream Verifier using HBK MicroStrain MIP SDK\n";
        std::cout << "==============================================\n";
        std::cout << "File                 : " << g_options.mip_path << "\n";
        std::cout << "Size                 : " << data.size() << " bytes\n";
        std::cout << "Expected frame id    : " << static_cast<int>(g_options.expected_frame_id) << "\n";
        std::cout << "Expected timebase    : "
                  << (g_options.check_timebase ? std::to_string(g_options.expected_timebase) : std::string("not checked"))
                  << "\n";

        scan_raw_mip_candidates(data);

        mip_parser parser;
        mip_parser_init(&parser, &packet_callback, nullptr, 100);
        const size_t bytes_consumed = mip_parser_parse(&parser, data.data(), data.size(), 0);
        if (bytes_consumed != data.size()) {
            std::cerr << "[WARN] Parser consumed " << bytes_consumed
                      << " of " << data.size() << " bytes.\n";
        }

        std::vector<Epoch> epochs = build_epochs_from_events();

        bool truth_used = false;
        if (!g_options.truth_csv_path.empty()) {
            std::vector<TruthRow> truth = load_truth_csv(g_options.truth_csv_path);
            truth_used = true;
            std::cout << "Truth CSV            : " << g_options.truth_csv_path << "\n";
            std::cout << "Truth rows           : " << truth.size() << "\n";
            compare_against_truth(epochs, truth);
        } else {
            std::cout << "Truth CSV            : not provided\n";
            std::cout << "Note                 : exact GPS comparison requires a truth CSV.\n";
        }

        print_sample_epochs(epochs);
        print_summary(epochs, truth_used);

        if (overall_ok(truth_used)) {
            if (truth_used) {
                std::cout << "[OK] MIP stream is valid and matches the GPS truth CSV within tolerances.\n";
            } else {
                std::cout << "[OK] MIP stream is valid and matches the expected main.cpp packet structure.\n";
                std::cout << "[INFO] GPS value equality was not checked because no truth CSV was provided.\n";
            }
            return 0;
        }

        std::cout << "[FAIL] Issues detected. Review the counters and messages above.\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
}
