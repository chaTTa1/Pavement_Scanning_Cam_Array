/*
 * Teensy 4.1  SBF/NMEA -> CV7 Bridge  (official MicroStrain MIP SDK)
 * ====================================================================
 * Hardware:
 *   Septentrio mosaic-H SBF or NMEA UART TX --> Teensy Pin 0 (Serial1 RX)
 *   Septentrio mosaic-H PPS output   --> Teensy Pin 2 (PPS_IN_PIN)
 *   Common GND                       --- Teensy GND / CV7 GND
 *
 *   Teensy Pin 8 (Serial2 TX) --> CV7 C-Series UUT_RX / RxD
 *   Teensy Pin 7 (Serial2 RX) <-- CV7 C-Series UUT_TX / TxD
 *   Teensy Pin 3 (PPS_OUT_IMU_PIN) --> CV7 GPIO1 (configure as PPS input in GUI)
 *   CV7 power/readout via C-Series USB.
 *
 * MIP descriptor set: 0x13 (Aiding)
 *   0x22  External Position LLH            (10 Hz from GNSS)
 *   0x29  External Velocity NED            (10 Hz from GNSS)
 *   0x31  External Heading True            (10 Hz from mosaic-H AttEuler, if valid)
 * System data set: 0xA0
 *   0x02  Time Sync Status                  (read-only aiding safety gate)
 *
 * CV7 configuration is intentionally not changed by this firmware. Configure
 * UART protocols, GPIO1/PPS, aiding sources, antenna/frame offsets, filter
 * initialization, and the Time Sync Status output stream with the CV7 GUI.
 *
 * Input auto-detection:
 *   - A valid $@ SBF block (length + CRC) locks this boot to SBF mode.
 *   - A '$' followed by any byte other than '@' locks this boot to NMEA mode.
 *   - Once NMEA is detected, every byte is forwarded unchanged to Serial2;
 *     NMEA sentence type, length, checksum, and line endings are not checked.
 */

#include <Arduino.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "mip/mip_all.h"
#include "mip/definitions/commands_aiding.h"
#include "mip/definitions/commands_base.h"
#include "mip/definitions/data_system.h"

using namespace mip::C;
using namespace microstrain::C;

// =========================================================================
// CONFIG
// =========================================================================
static constexpr uint32_t USB_BAUD  = 115200;
static constexpr uint32_t GNSS_BAUD = 115200;
static constexpr uint32_t IMU_BAUD  = 115200;

// DEBUG_MODE sends readable diagnostics to USB instead of a binary MIP mirror.
// Set false when you want record_mip.ps1 to capture generated MIP packets.
static constexpr bool DEBUG_MODE = true;
static constexpr bool MIRROR_MIP_TO_USB = !DEBUG_MODE;
static constexpr bool PRINT_STATUS_TO_USB = DEBUG_MODE;
static constexpr bool PRINT_EACH_CV7_ACK = true;

// PPS forwarding:
//   GPS 1 Hz PPS -> PPS_IN_PIN
//   PPS_OUT_IMU_PIN -> CV7 PPS input
// Change these two pins to match your wiring.
static constexpr uint8_t PPS_IN_PIN = 2;
static constexpr uint8_t PPS_OUT_IMU_PIN = 3;
// Teensy 4.1 built-in LED is on pin 13. It flashes once per GPS PPS rising edge.
static constexpr uint8_t PPS_LED_PIN = 13;
static constexpr uint32_t PPS_LED_FLASH_US = 200000;  // 200 ms, easy to see at 1 Hz.
static constexpr uint32_t GPS_TIME_UPDATE_MAX_PPS_DELAY_US = 250000;
static constexpr uint32_t CV7_TIME_SYNC_STATUS_MAX_AGE_MS = 2500;
static constexpr uint32_t GNSS_FRAME_INTERBYTE_TIMEOUT_MS = 250;
static constexpr uint32_t GNSS_FRAME_MAX_DURATION_MS = 500;

// CV7/GV7 SDK mip_time format used by Aiding commands:
//   timebase u8, reserved u8 (=1), nanoseconds u64.
// EXTERNAL_TIME tells the CV7 to interpret aiding timestamps against the
// external GPS/PPS timebase configured with GPS Time Update (0x01,0x72).
static constexpr mip_time_timebase AIDING_TIMEBASE = MIP_TIME_TIMEBASE_EXTERNAL_TIME;
static constexpr bool USE_GPS_TIME_NANOSECONDS = true;
static constexpr uint32_t INVALID_GNSS_TOW_MS = 0xFFFFFFFFu;
static constexpr uint16_t INVALID_GNSS_WEEK = 0xFFFFu;
static constexpr uint32_t GNSS_WEEK_LENGTH_MS = 604800000u;

// Keep external GNSS velocity aiding enabled, but mirror the exact aiding values
// sent to the CV7 as CSV over USB debug output for audit/comparison with NMEA.
static constexpr bool SEND_VELOCITY_AIDING = true;
static constexpr bool PRINT_AIDING_CSV_TO_USB = DEBUG_MODE;

// Set true only after confirming that mosaic-H AttEuler (SBF) or HDT (NMEA)
// is the vehicle +X true heading, including the dual-antenna baseline direction
// and any Septentrio attitude offset. Lever-arm validation alone is not enough.
static constexpr bool EXTERNAL_HEADING_FRAME_CONFIGURED = false;

// AUTO locks to SBF after the first CRC-valid SBF block, or to NMEA as soon as
// a '$' is followed by a byte other than '@'.
// This deliberately prevents the same GNSS solution from entering the filter
// once as converted MIP aiding and again as NMEA aiding.
enum class GnssInputProtocol : uint8_t { AUTO, SBF, NMEA };
static constexpr GnssInputProtocol GNSS_INPUT_PROTOCOL = GnssInputProtocol::AUTO;

// CV7 aiding frame ID. The CV7 supports frame IDs 1..4.
static constexpr uint8_t AIDING_FRAME_ID = 1;
static_assert(AIDING_FRAME_ID >= 1 && AIDING_FRAME_ID <= 4,
              "CV7 aiding frame ID must be in the range 1..4");

static uint8_t serial1_rx_buf[4096];

static volatile uint32_t pps_count = 0;
static volatile uint32_t last_pps_us = 0;
static volatile bool pps_led_active = false;
static volatile uint32_t pps_led_off_us = 0;

// =========================================================================
// SBF parser for the Septentrio mosaic-H blocks used by this bridge.
// =========================================================================
static constexpr uint16_t SBF_PVT_GEODETIC     = 4007;
static constexpr uint16_t SBF_POS_COV_GEODETIC = 5906;
static constexpr uint16_t SBF_VEL_COV_GEODETIC = 5908;
static constexpr uint16_t SBF_ATT_EULER        = 5938;
static constexpr uint16_t SBF_ATT_COV_EULER    = 5939;

static uint16_t sbf_crc16(const uint8_t* data, size_t len) {
    uint16_t crc = 0;
    for (size_t i = 0; i < len; ++i) {
        crc = (uint16_t)(crc ^ ((uint16_t)data[i] << 8));
        for (int j = 0; j < 8; ++j) {
            crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021)
                                 : (uint16_t)(crc << 1);
        }
    }
    return crc;
}

static inline bool is_dnu_d(double v) { return v <= -1.999e10; }
static inline bool is_dnu_f(float  v) { return v <= -1.999e9f; }

static bool finite_measurement(double v) { return isfinite(v) && !is_dnu_d(v); }
static bool finite_measurement(float v)  { return isfinite(v) && !is_dnu_f(v); }

static bool positive_finite_variance(float variance) {
    return isfinite(variance) && !is_dnu_f(variance) && variance > 0.0f;
}

static bool gnss_time_valid(uint16_t week, uint32_t tow_ms) {
    if (week == INVALID_GNSS_WEEK || tow_ms == INVALID_GNSS_TOW_MS
            || tow_ms >= GNSS_WEEK_LENGTH_MS) {
        return false;
    }

    static constexpr uint64_t NS_PER_MS = 1000000ULL;
    static constexpr uint64_t NS_PER_WEEK = 604800000000000ULL;
    const uint64_t tow_ns = (uint64_t)tow_ms * NS_PER_MS;
    return (uint64_t)week <= (UINT64_MAX - tow_ns) / NS_PER_WEEK;
}

static bool pvt_solution_mode_allowed(uint8_t mode) {
    switch (mode) {
        case 1:   // Stand-alone
        case 2:   // Differential
        case 4:   // RTK fixed
        case 5:   // RTK float
        case 6:   // SBAS
        case 7:   // Moving-base RTK fixed
        case 8:   // Moving-base RTK float
        case 10:  // PPP
            return true;
        default:
            return false;  // Includes no-PVT, fixed-location, and reserved modes.
    }
}

static uint64_t gps_time_ns(uint16_t week, uint32_t tow_ms) {
    static constexpr uint64_t NS_PER_MS = 1000000ULL;
    static constexpr uint64_t NS_PER_WEEK = 604800000000000ULL;
    return ((uint64_t)week * NS_PER_WEEK) + ((uint64_t)tow_ms * NS_PER_MS);
}

static float heading_deg_to_rad(float heading_deg) {
    float heading_rad = heading_deg * (float)(M_PI / 180.0);
    while (heading_rad > (float)M_PI) heading_rad -= (float)(2.0 * M_PI);
    while (heading_rad < (float)-M_PI) heading_rad += (float)(2.0 * M_PI);
    return heading_rad;
}

static void pps_isr() {
    const bool level = digitalReadFast(PPS_IN_PIN);
    digitalWriteFast(PPS_OUT_IMU_PIN, level);

    if (level) {
        const uint32_t now_us = micros();
        pps_count++;
        last_pps_us = now_us;
        pps_led_off_us = now_us + PPS_LED_FLASH_US;
        pps_led_active = true;
        digitalWriteFast(PPS_LED_PIN, HIGH);
    }
}

static void service_pps_led() {
    const uint32_t now_us = micros();
    noInterrupts();
    if (pps_led_active && (int32_t)(now_us - pps_led_off_us) >= 0) {
        digitalWriteFast(PPS_LED_PIN, LOW);
        pps_led_active = false;
    }
    interrupts();
}

struct PvtSnapshot {
    bool     valid = false;
    uint32_t tow_ms = 0;
    uint16_t week = 0;
    uint8_t  mode_raw = 0;
    uint8_t  mode = 0;
    bool     is_2d = false;
    uint8_t  err = 0;
    uint8_t  datum = 0xFF;
    bool     lat_valid = false, lon_valid = false, h_valid = false;
    bool     vn_valid = false, ve_valid = false, vu_valid = false;
    double   lat_deg = 0, lon_deg = 0, h_m = 0;
    float    vn = 0, ve = 0, vu = 0;
    uint8_t  num_sats = 0;
};
struct PosCovSnap {
    bool valid = false; uint32_t tow_ms = 0; uint16_t week = 0;
    uint8_t mode_raw = 0, mode = 0, err = 0; bool is_2d = false;
    bool valid_n = false, valid_e = false, valid_u = false;
    float std_n = 0, std_e = 0, std_u = 0;
};
struct VelCovSnap {
    bool valid = false; uint32_t tow_ms = 0; uint16_t week = 0;
    uint8_t mode_raw = 0, mode = 0, err = 0; bool is_2d = false;
    bool valid_vn = false, valid_ve = false, valid_vu = false;
    float std_vn = 0, std_ve = 0, std_vu = 0;
};
struct AttSnap {
    bool valid = false; uint32_t tow_ms = 0; uint16_t week = 0;
    uint8_t nr_sv = 0, err = 0;
    uint16_t mode = 0;
    float heading_deg = 0, pitch_deg = 0, roll_deg = 0;
};
struct AttCovSnap {
    bool valid = false;
    uint32_t tow_ms = 0; uint16_t week = 0;
    uint8_t err = 0;
    bool valid_heading = false, valid_pitch = false, valid_roll = false;
    float std_heading = 0, std_pitch = 0, std_roll = 0;
};

static PvtSnapshot pvt;
static PosCovSnap  pcv;
static VelCovSnap  vcv;
static AttSnap     att;
static AttCovSnap  acv;

static uint32_t cnt_pvt = 0, cnt_poscov = 0, cnt_velcov = 0, cnt_att = 0, cnt_attcov = 0;
static uint32_t mip_total = 0, mip_skipped = 0;
static uint16_t last_mip_week = INVALID_GNSS_WEEK;
static uint32_t last_mip_tow = INVALID_GNSS_TOW_MS;
static uint16_t last_heading_mip_week = INVALID_GNSS_WEEK;
static uint32_t last_heading_mip_tow = INVALID_GNSS_TOW_MS;
static uint16_t last_gps_time_update_week = INVALID_GNSS_WEEK;
static uint32_t last_gps_time_update_tow_s = 0xFFFFFFFFu;
static uint32_t last_gps_time_update_pps_count = 0;
static uint32_t last_gps_time_update_delay_us = 0xFFFFFFFFu;
static uint16_t last_pos_valid_flags = 0;
static uint16_t last_vel_valid_flags = 0;
static GnssInputProtocol active_input_protocol = GNSS_INPUT_PROTOCOL;
static mip_dispatch_handler cv7_time_sync_handler;
static mip_system_time_sync_status_data cv7_time_sync_status = {};
static bool cv7_time_sync_status_received = false;
static uint32_t cv7_time_sync_status_ms = 0;

static void print_aiding_csv_row(const mip_time& t, uint16_t pos_flags, uint16_t vel_flags);

static uint32_t cv7_ack_total = 0;
static uint32_t cv7_nack_total = 0;
static uint32_t cv7_ack_pos_llh = 0;
static uint32_t cv7_ack_vel_ned = 0;
static uint32_t cv7_ack_heading = 0;
static uint32_t cv7_ack_gps_time = 0;
static uint8_t last_cv7_ack_set = 0;
static uint8_t last_cv7_ack_field = 0;
static uint8_t last_cv7_ack_result = 0xFF;

static uint32_t dbg_serial1_bytes = 0;
static uint32_t dbg_sbf_sync = 0;
static uint32_t dbg_sbf_len_fail = 0;
static uint32_t dbg_sbf_oversize_drop = 0;
static uint32_t dbg_sbf_crc_ok = 0;
static uint32_t dbg_sbf_crc_fail = 0;
static uint32_t dbg_unknown_block = 0;
static uint32_t nmea_forwarded_bytes = 0;
static uint32_t dbg_gnss_frame_timeout = 0;
static uint32_t dbg_protocol_conflict = 0;

static uint32_t dbg_skip_no_pvt = 0;
static uint32_t dbg_skip_no_poscov = 0;
static uint32_t dbg_skip_no_velcov = 0;
static uint32_t dbg_skip_tow_mismatch = 0;
static uint32_t dbg_skip_week_mismatch = 0;
static uint32_t dbg_skip_time_not_synced = 0;
static uint32_t dbg_skip_non_wgs84 = 0;
static uint32_t dbg_invalid_mode_output = 0;
static uint32_t dbg_skip_repeat_tow = 0;

static uint32_t dbg_serializer_fail = 0;
static uint32_t dbg_serial2_write_fail = 0;

// =========================================================================
// CV7 ACK/NACK sniffer
// =========================================================================
static uint8_t cv7_packet_buf[MIP_PACKET_LENGTH_MAX];
static size_t cv7_packet_len = 0;
static size_t cv7_packet_expected_len = 0;

static bool mip_checksum_ok(const uint8_t* packet, size_t length) {
    if (length < 6) return false;
    uint8_t a = 0;
    uint8_t b = 0;
    for (size_t i = 0; i < length - 2; ++i) {
        a = (uint8_t)(a + packet[i]);
        b = (uint8_t)(b + a);
    }
    return a == packet[length - 2] && b == packet[length - 1];
}

static const char* ack_result_name(uint8_t result) {
    switch (result) {
        case 0x00: return "ACK_OK";
        case 0x01: return "NACK_UNKNOWN";
        case 0x02: return "NACK_CHECKSUM";
        case 0x03: return "NACK_PARAM";
        case 0x04: return "NACK_FAILED";
        case 0x05: return "NACK_TIMEOUT";
        default: return "UNKNOWN";
    }
}

static const char* command_name(uint8_t descriptor_set, uint8_t field_descriptor) {
    if (descriptor_set == MIP_AIDING_CMD_DESC_SET) {
        switch (field_descriptor) {
            case MIP_CMD_DESC_AIDING_FRAME_CONFIG: return "FRAME_CONFIG";
            case MIP_CMD_DESC_AIDING_POS_LLH: return "POS_LLH";
            case MIP_CMD_DESC_AIDING_VEL_NED: return "VEL_NED";
            case MIP_CMD_DESC_AIDING_HEADING_TRUE: return "HEADING_TRUE";
            default: return "AIDING";
        }
    }
    if (descriptor_set == MIP_BASE_CMD_DESC_SET &&
        field_descriptor == MIP_CMD_DESC_BASE_GPS_TIME_UPDATE) {
        return "GPS_TIME_UPDATE";
    }
    return "COMMAND";
}

static void record_cv7_ack(uint8_t descriptor_set, uint8_t field_descriptor, uint8_t result) {
    last_cv7_ack_set = descriptor_set;
    last_cv7_ack_field = field_descriptor;
    last_cv7_ack_result = result;

    if (result == MIP_ACK_OK) {
        cv7_ack_total++;
        if (descriptor_set == MIP_AIDING_CMD_DESC_SET && field_descriptor == MIP_CMD_DESC_AIDING_POS_LLH) {
            cv7_ack_pos_llh++;
        } else if (descriptor_set == MIP_AIDING_CMD_DESC_SET && field_descriptor == MIP_CMD_DESC_AIDING_VEL_NED) {
            cv7_ack_vel_ned++;
        } else if (descriptor_set == MIP_AIDING_CMD_DESC_SET && field_descriptor == MIP_CMD_DESC_AIDING_HEADING_TRUE) {
            cv7_ack_heading++;
        } else if (descriptor_set == MIP_BASE_CMD_DESC_SET && field_descriptor == MIP_CMD_DESC_BASE_GPS_TIME_UPDATE) {
            cv7_ack_gps_time++;
        }
    } else {
        cv7_nack_total++;
    }

    if (PRINT_STATUS_TO_USB && PRINT_EACH_CV7_ACK) {
        Serial.print(F("[cv7] "));
        Serial.print(command_name(descriptor_set, field_descriptor));
        Serial.print(F(" ack ds=0x"));
        Serial.print(descriptor_set, HEX);
        Serial.print(F(" field=0x"));
        Serial.print(field_descriptor, HEX);
        Serial.print(F(" result=0x"));
        Serial.print(result, HEX);
        Serial.print(F(" "));
        Serial.println(ack_result_name(result));
    }
}

static void inspect_cv7_packet_for_acks(const uint8_t* packet, size_t length) {
    if (length < 6 || packet[0] != 0x75 || packet[1] != 0x65 || !mip_checksum_ok(packet, length)) {
        return;
    }

    const uint8_t descriptor_set = packet[2];
    const uint8_t payload_length = packet[3];
    size_t pos = 4;
    const size_t payload_end = 4 + payload_length;

    while (pos + 2 <= payload_end) {
        const uint8_t field_length = packet[pos];
        const uint8_t field_descriptor = packet[pos + 1];
        if (field_length < 2 || pos + field_length > payload_end) {
            break;
        }
        if (field_descriptor == 0xF1 && field_length >= 4) {
            const uint8_t acked_field = packet[pos + 2];
            const uint8_t result = packet[pos + 3];
            record_cv7_ack(descriptor_set, acked_field, result);
        }
        pos += field_length;
    }
}

static void feed_cv7_ack_sniffer(uint8_t byte) {
    if (cv7_packet_len == 0) {
        if (byte == 0x75) {
            cv7_packet_buf[cv7_packet_len++] = byte;
        }
        return;
    }

    if (cv7_packet_len == 1) {
        if (byte == 0x65) {
            cv7_packet_buf[cv7_packet_len++] = byte;
        } else {
            cv7_packet_len = (byte == 0x75) ? 1 : 0;
            if (cv7_packet_len == 1) cv7_packet_buf[0] = byte;
        }
        return;
    }

    if (cv7_packet_len >= sizeof(cv7_packet_buf)) {
        cv7_packet_len = 0;
        cv7_packet_expected_len = 0;
        return;
    }

    cv7_packet_buf[cv7_packet_len++] = byte;

    if (cv7_packet_len == 4) {
        cv7_packet_expected_len = 4 + cv7_packet_buf[3] + 2;
        if (cv7_packet_expected_len > sizeof(cv7_packet_buf)) {
            cv7_packet_len = 0;
            cv7_packet_expected_len = 0;
        }
    }

    if (cv7_packet_expected_len > 0 && cv7_packet_len == cv7_packet_expected_len) {
        inspect_cv7_packet_for_acks(cv7_packet_buf, cv7_packet_len);
        cv7_packet_len = 0;
        cv7_packet_expected_len = 0;
    }
}

static void parse_pvt_geodetic(const uint8_t* blk, uint16_t len) {
    // Datum and NrSV are at offsets 73 and 74 in PVTGeodetic revision 2.
    if (len < 75) {
        dbg_sbf_len_fail++;
        return;
    }

    PvtSnapshot s;
    memcpy(&s.tow_ms, blk + 8, 4);
    memcpy(&s.week,   blk + 12, 2);
    s.mode_raw = blk[14];
    s.mode = s.mode_raw & 0x0F;
    s.is_2d = (s.mode_raw & 0x80u) != 0;
    s.err  = blk[15];
    double lat_rad, lon_rad, h;
    memcpy(&lat_rad, blk + 16, 8);
    memcpy(&lon_rad, blk + 24, 8);
    memcpy(&h,       blk + 32, 8);
    s.lat_deg = lat_rad * (180.0 / M_PI);
    s.lon_deg = lon_rad * (180.0 / M_PI);
    s.h_m = h;
    memcpy(&s.vn, blk + 44, 4);
    memcpy(&s.ve, blk + 48, 4);
    memcpy(&s.vu, blk + 52, 4);
    s.datum = blk[73];
    s.num_sats = blk[74];
    s.lat_valid = finite_measurement(s.lat_deg) && s.lat_deg >= -90.0 && s.lat_deg <= 90.0;
    s.lon_valid = finite_measurement(s.lon_deg) && s.lon_deg >= -180.0 && s.lon_deg <= 180.0;
    s.h_valid = finite_measurement(s.h_m);
    s.vn_valid = finite_measurement(s.vn);
    s.ve_valid = finite_measurement(s.ve);
    s.vu_valid = finite_measurement(s.vu);
    s.valid = s.err == 0 && (s.mode_raw & 0x70u) == 0
           && pvt_solution_mode_allowed(s.mode) && s.datum == 0
           && (s.lat_valid || s.lon_valid || s.h_valid
               || s.vn_valid || s.ve_valid || s.vu_valid);
    pvt = s;
    cnt_pvt++;
}

static void parse_pos_cov_geodetic(const uint8_t* blk, uint16_t len) {
    if (len < 28) {
        dbg_sbf_len_fail++;
        return;
    }

    PosCovSnap s;
    memcpy(&s.tow_ms, blk + 8, 4);
    memcpy(&s.week, blk + 12, 2);
    s.mode_raw = blk[14];
    s.mode = s.mode_raw & 0x0F;
    s.is_2d = (s.mode_raw & 0x80u) != 0;
    s.err = blk[15];
    float var_n, var_e, var_u;
    memcpy(&var_n, blk + 16, 4);
    memcpy(&var_e, blk + 20, 4);
    memcpy(&var_u, blk + 24, 4);
    s.valid_n = positive_finite_variance(var_n);
    s.valid_e = positive_finite_variance(var_e);
    s.valid_u = positive_finite_variance(var_u) && !s.is_2d;
    s.std_n = s.valid_n ? sqrtf(var_n) : 0.0f;
    s.std_e = s.valid_e ? sqrtf(var_e) : 0.0f;
    s.std_u = s.valid_u ? sqrtf(var_u) : 0.0f;
    s.valid_n = s.valid_n && isfinite(s.std_n) && s.std_n > 0.0f;
    s.valid_e = s.valid_e && isfinite(s.std_e) && s.std_e > 0.0f;
    s.valid_u = s.valid_u && isfinite(s.std_u) && s.std_u > 0.0f;
    s.valid = s.err == 0 && (s.mode_raw & 0x70u) == 0
           && pvt_solution_mode_allowed(s.mode)
           && (s.valid_n || s.valid_e || s.valid_u);
    pcv = s;
    cnt_poscov++;
}

static void parse_vel_cov_geodetic(const uint8_t* blk, uint16_t len) {
    if (len < 28) {
        dbg_sbf_len_fail++;
        return;
    }

    VelCovSnap s;
    memcpy(&s.tow_ms, blk + 8, 4);
    memcpy(&s.week, blk + 12, 2);
    s.mode_raw = blk[14];
    s.mode = s.mode_raw & 0x0F;
    s.is_2d = (s.mode_raw & 0x80u) != 0;
    s.err = blk[15];
    float var_vn, var_ve, var_vu;
    memcpy(&var_vn, blk + 16, 4);
    memcpy(&var_ve, blk + 20, 4);
    memcpy(&var_vu, blk + 24, 4);
    s.valid_vn = positive_finite_variance(var_vn);
    s.valid_ve = positive_finite_variance(var_ve);
    s.valid_vu = positive_finite_variance(var_vu) && !s.is_2d;
    s.std_vn = s.valid_vn ? sqrtf(var_vn) : 0.0f;
    s.std_ve = s.valid_ve ? sqrtf(var_ve) : 0.0f;
    s.std_vu = s.valid_vu ? sqrtf(var_vu) : 0.0f;
    s.valid_vn = s.valid_vn && isfinite(s.std_vn) && s.std_vn > 0.0f;
    s.valid_ve = s.valid_ve && isfinite(s.std_ve) && s.std_ve > 0.0f;
    s.valid_vu = s.valid_vu && isfinite(s.std_vu) && s.std_vu > 0.0f;
    s.valid = s.err == 0 && (s.mode_raw & 0x70u) == 0
           && pvt_solution_mode_allowed(s.mode)
           && (s.valid_vn || s.valid_ve || s.valid_vu);
    vcv = s;
    cnt_velcov++;
}

static void parse_att_euler(const uint8_t* blk, uint16_t len) {
    if (len < 32) {
        dbg_sbf_len_fail++;
        return;
    }

    AttSnap s;
    memcpy(&s.tow_ms, blk + 8, 4);
    memcpy(&s.week, blk + 12, 2);
    s.nr_sv = blk[14];
    s.err = blk[15];
    memcpy(&s.mode, blk + 16, 2);
    memcpy(&s.heading_deg, blk + 20, 4);
    memcpy(&s.pitch_deg,   blk + 24, 4);
    memcpy(&s.roll_deg,    blk + 28, 4);
    s.valid = (s.err == 0) && (s.mode >= 1 && s.mode <= 4)
           && finite_measurement(s.heading_deg);
    att = s;
    cnt_att++;
}

static void parse_att_cov_euler(const uint8_t* blk, uint16_t len) {
    if (len < 28) {
        dbg_sbf_len_fail++;
        return;
    }

    AttCovSnap s;
    memcpy(&s.tow_ms, blk + 8, 4);
    memcpy(&s.week, blk + 12, 2);
    s.err = blk[15];
    float var_h, var_p, var_r;
    memcpy(&var_h, blk + 16, 4);
    memcpy(&var_p, blk + 20, 4);
    memcpy(&var_r, blk + 24, 4);
    s.valid_heading = positive_finite_variance(var_h);
    s.valid_pitch = positive_finite_variance(var_p);
    s.valid_roll = positive_finite_variance(var_r);
    s.std_heading = s.valid_heading ? sqrtf(var_h) : 0.0f;
    s.std_pitch   = s.valid_pitch ? sqrtf(var_p) : 0.0f;
    s.std_roll    = s.valid_roll ? sqrtf(var_r) : 0.0f;
    s.valid_heading = s.valid_heading && isfinite(s.std_heading) && s.std_heading > 0.0f;
    s.valid_pitch = s.valid_pitch && isfinite(s.std_pitch) && s.std_pitch > 0.0f;
    s.valid_roll = s.valid_roll && isfinite(s.std_roll) && s.std_roll > 0.0f;
    s.valid = (s.err == 0) && s.valid_heading;
    acv = s;
    cnt_attcov++;
}

// =========================================================================
// MIP SDK interface adapter
// =========================================================================
static mip_interface mip_dev;

static bool app_mip_send_callback(mip_interface* /*device*/,
                                  const uint8_t* data, size_t length) {
    size_t written = Serial2.write(data, length);
    Serial2.flush();
    if (written != length) {
        dbg_serial2_write_fail++;
        return false;
    }
    return true;
}

static bool app_mip_recv_callback(mip_interface* /*device*/,
                                  uint8_t* buffer, size_t max_length,
                                  mip_timeout /*wait_time*/,
                                  bool /*from_cmd*/,
                                  size_t* out_length,
                                  mip_timestamp* timestamp_out) {
    size_t n = 0;
    while (Serial2.available() && n < max_length) {
        int b = Serial2.read();
        if (b < 0) break;
        const uint8_t byte = (uint8_t)b;
        feed_cv7_ack_sniffer(byte);
        buffer[n++] = byte;
    }
    *out_length = n;
    *timestamp_out = (mip_timestamp)millis();
    return true;
}

static bool app_mip_update_callback(mip_interface* device,
                                    mip_timeout wait_time,
                                    bool from_cmd) {
    return mip_interface_default_update(device, wait_time, from_cmd);
}

static bool mip_send_command_field(uint8_t field_descriptor,
                                   const uint8_t* payload,
                                   size_t payload_length,
                                   uint8_t descriptor_set = MIP_AIDING_CMD_DESC_SET) {
    if (payload_length > MIP_FIELD_PAYLOAD_LENGTH_MAX) {
        return false;
    }

    uint8_t packet_buffer[MIP_PACKET_LENGTH_MAX];
    mip_packet_view packet;
    mip_packet_create(&packet, packet_buffer, sizeof(packet_buffer), descriptor_set);
    if (!mip_packet_add_field(&packet, field_descriptor, payload, (uint8_t)payload_length)) {
        return false;
    }
    mip_packet_finalize(&packet);

    const uint8_t* packet_data = mip_packet_data(&packet);
    const size_t packet_length = mip_packet_total_length(&packet);
    const bool sent = mip_interface_send_to_device(&mip_dev, packet_data, packet_length);

    if (sent && MIRROR_MIP_TO_USB) {
        Serial.write(packet_data, packet_length);
    }

    return sent;
}

static void handle_cv7_time_sync_status(void*, const mip_field_view* field,
                                        mip_timestamp) {
    mip_system_time_sync_status_data status = {};
    if (extract_mip_system_time_sync_status_data_from_field(field, &status)) {
        cv7_time_sync_status = status;
        cv7_time_sync_status_received = true;
        cv7_time_sync_status_ms = millis();
    }
}

static bool cv7_time_sync_ready() {
    return cv7_time_sync_status_received
        && cv7_time_sync_status.time_sync
        && cv7_time_sync_status.last_pps_rcvd <= 1
        && (uint32_t)(millis() - cv7_time_sync_status_ms)
                <= CV7_TIME_SYNC_STATUS_MAX_AGE_MS;
}

static void maybe_send_gps_time_update(uint16_t week, uint32_t tow_ms) {
    // The week/TOW command identifies the rising PPS edge. Only send it while
    // still inside the CV7-INS 250 ms acceptance window for that edge.
    if (active_input_protocol != GnssInputProtocol::SBF
            || !gnss_time_valid(week, tow_ms)) {
        return;
    }

    uint32_t pps_snapshot;
    uint32_t pps_us_snapshot;
    noInterrupts();
    pps_snapshot = pps_count;
    pps_us_snapshot = last_pps_us;
    interrupts();

    if (pps_snapshot == 0 || pps_snapshot == last_gps_time_update_pps_count) {
        return;
    }

    const uint32_t elapsed_us = (uint32_t)(micros() - pps_us_snapshot);
    if (elapsed_us > GPS_TIME_UPDATE_MAX_PPS_DELAY_US) {
        return;
    }

    // At 10 Hz, the first post-PPS PVT can be 0, 100, or 200 ms into the
    // same GPS second. A larger remainder may be a delayed pre-PPS epoch.
    if ((tow_ms % 1000u) > (GPS_TIME_UPDATE_MAX_PPS_DELAY_US / 1000u)) {
        return;
    }

    const uint32_t tow_s = tow_ms / 1000u;
    if (week == last_gps_time_update_week && tow_s == last_gps_time_update_tow_s) {
        last_gps_time_update_pps_count = pps_snapshot;
        return;
    }

    const mip_cmd_result tow_result = mip_base_write_gps_time_update(
        &mip_dev, MIP_BASE_GPS_TIME_UPDATE_COMMAND_FIELD_ID_TIME_OF_WEEK, tow_s);
    if (!mip_cmd_result_is_ack(tow_result)) {
        return;
    }

    // The SDK command waits for ACK. Do not start the week command if that
    // wait consumed the remainder of the PPS acceptance window.
    const uint32_t elapsed_after_tow_us = (uint32_t)(micros() - pps_us_snapshot);
    if (elapsed_after_tow_us > GPS_TIME_UPDATE_MAX_PPS_DELAY_US) {
        return;
    }

    const mip_cmd_result week_result = mip_base_write_gps_time_update(
        &mip_dev, MIP_BASE_GPS_TIME_UPDATE_COMMAND_FIELD_ID_WEEK_NUMBER, week);

    if (mip_cmd_result_is_ack(week_result)) {
        last_gps_time_update_week = week;
        last_gps_time_update_tow_s = tow_s;
        last_gps_time_update_pps_count = pps_snapshot;
        // Approximate the transmission start of the second command. Waiting
        // for its ACK is not part of the CV7's 250 ms receive window.
        last_gps_time_update_delay_us = elapsed_after_tow_us;
    }
}

static const char* input_protocol_name(GnssInputProtocol protocol) {
    switch (protocol) {
        case GnssInputProtocol::SBF:  return "SBF -> MIP";
        case GnssInputProtocol::NMEA: return "NMEA passthrough";
        default:                      return "AUTO detecting";
    }
}

static bool select_input_protocol(GnssInputProtocol detected) {
    if (active_input_protocol == GnssInputProtocol::AUTO) {
        active_input_protocol = detected;
        if (PRINT_STATUS_TO_USB) {
            Serial.print(F("[mode] Current GNSS protocol: "));
            Serial.println(active_input_protocol == GnssInputProtocol::NMEA
                ? F("NMEA") : F("SBF"));
        }
    }

    if (active_input_protocol != detected) {
        dbg_protocol_conflict++;
        return false;
    }
    return true;
}

static bool pvt_mode_allows_valid_aiding() {
    return pvt.err == 0 && (pvt.mode_raw & 0x70u) == 0
        && pvt_solution_mode_allowed(pvt.mode);
}

static uint16_t position_valid_flags() {
    if (!pvt.valid || !pvt_mode_allows_valid_aiding() || pvt.datum != 0
            || !pcv.valid || pcv.err != 0 || (pcv.mode_raw & 0x70u) != 0
            || pcv.mode != pvt.mode || pcv.is_2d != pvt.is_2d) {
        return MIP_AIDING_POS_LLH_COMMAND_VALID_FLAGS_NONE;
    }

    uint16_t flags = MIP_AIDING_POS_LLH_COMMAND_VALID_FLAGS_NONE;
    if (pvt.lat_valid && pcv.valid_n)
        flags |= MIP_AIDING_POS_LLH_COMMAND_VALID_FLAGS_LATITUDE;
    if (pvt.lon_valid && pcv.valid_e)
        flags |= MIP_AIDING_POS_LLH_COMMAND_VALID_FLAGS_LONGITUDE;
    if (!pvt.is_2d && pvt.h_valid && pcv.valid_u)
        flags |= MIP_AIDING_POS_LLH_COMMAND_VALID_FLAGS_HEIGHT;
    return flags;
}

static uint16_t velocity_valid_flags() {
    if (!pvt.valid || !pvt_mode_allows_valid_aiding() || pvt.datum != 0
            || !vcv.valid || vcv.err != 0 || (vcv.mode_raw & 0x70u) != 0
            || vcv.mode != pvt.mode || vcv.is_2d != pvt.is_2d) {
        return MIP_AIDING_VEL_NED_COMMAND_VALID_FLAGS_NONE;
    }

    uint16_t flags = MIP_AIDING_VEL_NED_COMMAND_VALID_FLAGS_NONE;
    if (pvt.vn_valid && vcv.valid_vn)
        flags |= MIP_AIDING_VEL_NED_COMMAND_VALID_FLAGS_X;
    if (pvt.ve_valid && vcv.valid_ve)
        flags |= MIP_AIDING_VEL_NED_COMMAND_VALID_FLAGS_Y;
    if (!pvt.is_2d && pvt.vu_valid && vcv.valid_vu)
        flags |= MIP_AIDING_VEL_NED_COMMAND_VALID_FLAGS_Z;
    return flags;
}

// =========================================================================
// MIP REALTIME OUTPUT (position/velocity epochs plus independent heading epochs)
// =========================================================================
static void try_send_heading() {
    if (!EXTERNAL_HEADING_FRAME_CONFIGURED
            || !cv7_time_sync_ready()
            || cnt_att == 0 || cnt_attcov == 0) {
        return;
    }

    // AttEuler and AttCovEuler describe one measurement only when both parts
    // have the same complete GPS epoch. Do not pair adjacent 10 Hz samples.
    if (!gnss_time_valid(att.week, att.tow_ms)
            || !gnss_time_valid(acv.week, acv.tow_ms)
            || att.week != acv.week || att.tow_ms != acv.tow_ms
            || !att.valid || !acv.valid || !acv.valid_heading
            || !finite_measurement(att.heading_deg)
            || !isfinite(acv.std_heading) || acv.std_heading <= 0.0f
            || (att.week == last_heading_mip_week
                && att.tow_ms == last_heading_mip_tow)) {
        return;
    }

    mip_time heading_time;
    heading_time.timebase = AIDING_TIMEBASE;
    heading_time.reserved = 1;
    heading_time.nanoseconds = USE_GPS_TIME_NANOSECONDS
                             ? gps_time_ns(att.week, att.tow_ms) : 0ULL;

    mip_aiding_heading_true_command cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.time = heading_time;
    cmd.frame_id = AIDING_FRAME_ID;
    cmd.heading = heading_deg_to_rad(att.heading_deg);
    cmd.uncertainty = acv.std_heading * (float)(M_PI / 180.0);
    cmd.valid_flags = 0x0001;

    mip_serializer s;
    uint8_t buf[80];
    microstrain_serializer_init_insertion(&s, buf, sizeof(buf));
    insert_mip_aiding_heading_true_command(&s, &cmd);
    if (microstrain_serializer_is_ok(&s)) {
        if (mip_send_command_field(MIP_CMD_DESC_AIDING_HEADING_TRUE,
                                   buf, microstrain_serializer_length(&s))) {
            mip_total++;
            last_heading_mip_week = att.week;
            last_heading_mip_tow = att.tow_ms;
        }
    } else {
        dbg_serializer_fail++;
    }
}

static void try_send_mip() {
    if (cnt_pvt == 0) {
        dbg_skip_no_pvt++;
        mip_skipped++;
        return;
    }
    if (cnt_poscov == 0) {
        dbg_skip_no_poscov++;
        mip_skipped++;
        return;
    }
    if (cnt_velcov == 0) {
        dbg_skip_no_velcov++;
        mip_skipped++;
        return;
    }

    // Assemble by complete GPS epoch, independent of SBF block arrival order.
    // Do not consume an epoch until all three receiver timestamps match.
    const bool time_is_valid = gnss_time_valid(pvt.week, pvt.tow_ms)
                            && gnss_time_valid(pcv.week, pcv.tow_ms)
                            && gnss_time_valid(vcv.week, vcv.tow_ms);
    if (USE_GPS_TIME_NANOSECONDS && !time_is_valid) {
        mip_skipped++;
        return;
    }
    if (time_is_valid && (pvt.week != pcv.week || pvt.week != vcv.week)) {
        dbg_skip_week_mismatch++;
        return;
    }
    if (time_is_valid && (pvt.tow_ms != pcv.tow_ms || pvt.tow_ms != vcv.tow_ms)) {
        dbg_skip_tow_mismatch++;
        return;
    }
    if (time_is_valid && pvt.week == last_mip_week && pvt.tow_ms == last_mip_tow) {
        dbg_skip_repeat_tow++;
        return;
    }

    if (!cv7_time_sync_ready()) {
        dbg_skip_time_not_synced++;
        mip_skipped++;
        return;
    }
    if (pvt.datum != 0) {
        dbg_skip_non_wgs84++;
        mip_skipped++;
        return;
    }
    if (!pvt_mode_allows_valid_aiding()) {
        dbg_invalid_mode_output++;
        mip_skipped++;
        return;
    }

    const uint16_t pos_flags = position_valid_flags();
    const uint16_t vel_flags = velocity_valid_flags();
    if (pos_flags == MIP_AIDING_POS_LLH_COMMAND_VALID_FLAGS_NONE
            && (!SEND_VELOCITY_AIDING
                || vel_flags == MIP_AIDING_VEL_NED_COMMAND_VALID_FLAGS_NONE)) {
        mip_skipped++;
        return;
    }

    last_pos_valid_flags = pos_flags;
    last_vel_valid_flags = SEND_VELOCITY_AIDING ? vel_flags
                                                : MIP_AIDING_VEL_NED_COMMAND_VALID_FLAGS_NONE;
    bool sent_epoch_aiding = false;

    // Build mip_time once, reuse for all 3 fields
    mip_time t;
    t.timebase = AIDING_TIMEBASE;
    t.reserved = 1;
    t.nanoseconds = USE_GPS_TIME_NANOSECONDS ? gps_time_ns(pvt.week, pvt.tow_ms) : 0ULL;

    // --- Field 1: External Position LLH (0x22) ---
    if (pos_flags != MIP_AIDING_POS_LLH_COMMAND_VALID_FLAGS_NONE) {
        double position[3] = {
            (pos_flags & MIP_AIDING_POS_LLH_COMMAND_VALID_FLAGS_LATITUDE) ? pvt.lat_deg : 0.0,
            (pos_flags & MIP_AIDING_POS_LLH_COMMAND_VALID_FLAGS_LONGITUDE) ? pvt.lon_deg : 0.0,
            (pos_flags & MIP_AIDING_POS_LLH_COMMAND_VALID_FLAGS_HEIGHT) ? pvt.h_m : 0.0
        };
        float uncertainty[3] = {
            (pos_flags & MIP_AIDING_POS_LLH_COMMAND_VALID_FLAGS_LATITUDE) ? pcv.std_n : 0.0f,
            (pos_flags & MIP_AIDING_POS_LLH_COMMAND_VALID_FLAGS_LONGITUDE) ? pcv.std_e : 0.0f,
            (pos_flags & MIP_AIDING_POS_LLH_COMMAND_VALID_FLAGS_HEIGHT) ? pcv.std_u : 0.0f
        };

        mip_aiding_pos_llh_command cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.time = t;
        cmd.frame_id = AIDING_FRAME_ID;
        cmd.latitude  = position[0];
        cmd.longitude = position[1];
        cmd.height    = position[2];
        cmd.uncertainty[0] = uncertainty[0];
        cmd.uncertainty[1] = uncertainty[1];
        cmd.uncertainty[2] = uncertainty[2];
        cmd.valid_flags = pos_flags;
        // Send as one-shot field (no response expected)
        mip_serializer s;
        uint8_t buf[80];
        microstrain_serializer_init_insertion(&s, buf, sizeof(buf));
        insert_mip_aiding_pos_llh_command(&s, &cmd);
        if (microstrain_serializer_is_ok(&s)) {
            if (mip_send_command_field(MIP_CMD_DESC_AIDING_POS_LLH,
                                       buf, microstrain_serializer_length(&s))) {
                mip_total++;
                sent_epoch_aiding = true;
            }
        } else {
            dbg_serializer_fail++;
        }
    }

    // --- Field 2: External Velocity NED (0x29) ---
    if (SEND_VELOCITY_AIDING
            && vel_flags != MIP_AIDING_VEL_NED_COMMAND_VALID_FLAGS_NONE) {
        mip_aiding_vel_ned_command cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.time = t;
        cmd.frame_id = AIDING_FRAME_ID;
        cmd.velocity[0] = (vel_flags & MIP_AIDING_VEL_NED_COMMAND_VALID_FLAGS_X)
                        ? pvt.vn : 0.0f;       // North
        cmd.velocity[1] = (vel_flags & MIP_AIDING_VEL_NED_COMMAND_VALID_FLAGS_Y)
                        ? pvt.ve : 0.0f;       // East
        cmd.velocity[2] = (vel_flags & MIP_AIDING_VEL_NED_COMMAND_VALID_FLAGS_Z)
                        ? -pvt.vu : 0.0f;      // Down = -Up
        cmd.uncertainty[0] = (vel_flags & MIP_AIDING_VEL_NED_COMMAND_VALID_FLAGS_X)
                           ? vcv.std_vn : 0.0f;
        cmd.uncertainty[1] = (vel_flags & MIP_AIDING_VEL_NED_COMMAND_VALID_FLAGS_Y)
                           ? vcv.std_ve : 0.0f;
        cmd.uncertainty[2] = (vel_flags & MIP_AIDING_VEL_NED_COMMAND_VALID_FLAGS_Z)
                           ? vcv.std_vu : 0.0f;
        cmd.valid_flags = vel_flags;
        mip_serializer s;
        uint8_t buf[80];
        microstrain_serializer_init_insertion(&s, buf, sizeof(buf));
        insert_mip_aiding_vel_ned_command(&s, &cmd);
        if (microstrain_serializer_is_ok(&s)) {
            if (mip_send_command_field(MIP_CMD_DESC_AIDING_VEL_NED,
                                       buf, microstrain_serializer_length(&s))) {
                mip_total++;
                sent_epoch_aiding = true;
            }
        } else {
            dbg_serializer_fail++;
        }
    }

    print_aiding_csv_row(t, last_pos_valid_flags, last_vel_valid_flags);

    if (sent_epoch_aiding) {
        last_mip_week = pvt.week;
        last_mip_tow = pvt.tow_ms;
    }
}

static void handle_sbf_packet(uint16_t blk_id, const uint8_t* blk, uint16_t len) {
    blk_id &= 0x1FFF;
    switch (blk_id) {
        case SBF_PVT_GEODETIC: {
            const uint32_t previous_count = cnt_pvt;
            parse_pvt_geodetic(blk, len);
            if (cnt_pvt != previous_count) {
                // GPS Time Update is intentionally independent of the Time Sync
                // gate; it is what allows the CV7 to establish that sync.
                maybe_send_gps_time_update(pvt.week, pvt.tow_ms);
            }
            break;
        }
        case SBF_POS_COV_GEODETIC: parse_pos_cov_geodetic(blk, len); break;
        case SBF_VEL_COV_GEODETIC: parse_vel_cov_geodetic(blk, len); break;
        case SBF_ATT_EULER:        parse_att_euler(blk, len);        break;
        case SBF_ATT_COV_EULER:    parse_att_cov_euler(blk, len);    break;
        default:
            dbg_unknown_block++;
            return;
    }
    try_send_mip();
    try_send_heading();
}

// =========================================================================
// Serial1 SBF/NMEA demultiplexer.
// AUTO only holds the leading '$' until the next byte distinguishes SBF ('$@')
// from NMEA. After NMEA is selected, all subsequent bytes bypass this parser
// and are forwarded unchanged. SBF remains framed and CRC-validated.
// =========================================================================
static uint8_t sbf_buf[1024];
static uint16_t sbf_idx = 0;
static uint16_t sbf_len = 0;
static uint32_t sbf_drop_remaining = 0;
static uint32_t gnss_last_byte_ms = 0;
static uint32_t gnss_frame_start_ms = 0;

enum class GnssStreamState : uint8_t {
    SEEK_DOLLAR,
    AFTER_DOLLAR,
    SBF_BODY,
    DROP_SBF_REMAINDER
};
static GnssStreamState gnss_stream_state = GnssStreamState::SEEK_DOLLAR;

static void reset_gnss_stream_parser() {
    gnss_stream_state = GnssStreamState::SEEK_DOLLAR;
    sbf_idx = 0;
    sbf_len = 0;
    sbf_drop_remaining = 0;
}

static void begin_dollar_candidate() {
    sbf_buf[0] = '$';
    sbf_idx = 1;
    sbf_len = 0;
    gnss_frame_start_ms = millis();
    gnss_stream_state = GnssStreamState::AFTER_DOLLAR;
}

static void forward_nmea_bytes(const uint8_t* data, size_t length) {
    const size_t written = Serial2.write(data, length);
    nmea_forwarded_bytes += (uint32_t)written;
    if (written != length) {
        dbg_serial2_write_fail++;
    }
}

static void finish_sbf_candidate() {
    uint16_t crc_expect = 0;
    memcpy(&crc_expect, &sbf_buf[2], sizeof(crc_expect));
    const uint16_t crc_calc = sbf_crc16(&sbf_buf[4], sbf_len - 4);
    if (crc_calc != crc_expect) {
        dbg_sbf_crc_fail++;
        return;
    }

    dbg_sbf_crc_ok++;
    if (!select_input_protocol(GnssInputProtocol::SBF)) {
        return;
    }
    uint16_t blk_id = 0;
    memcpy(&blk_id, &sbf_buf[4], sizeof(blk_id));
    handle_sbf_packet(blk_id, sbf_buf, sbf_len);
}

static void feed_serial1_byte(uint8_t b) {
    dbg_serial1_bytes++;

    // Explicit NMEA mode, or AUTO after its first non-SBF '$x' prefix, is a
    // transparent byte stream. No sentence parsing or validation occurs here.
    if (active_input_protocol == GnssInputProtocol::NMEA) {
        forward_nmea_bytes(&b, 1);
        return;
    }

    const uint32_t now_ms = millis();
    if (gnss_stream_state != GnssStreamState::SEEK_DOLLAR
            && ((uint32_t)(now_ms - gnss_last_byte_ms)
                    > GNSS_FRAME_INTERBYTE_TIMEOUT_MS
                || (gnss_stream_state != GnssStreamState::DROP_SBF_REMAINDER
                    && (uint32_t)(now_ms - gnss_frame_start_ms)
                        > GNSS_FRAME_MAX_DURATION_MS))) {
        dbg_gnss_frame_timeout++;
        reset_gnss_stream_parser();
    }
    gnss_last_byte_ms = now_ms;

    switch (gnss_stream_state) {
        case GnssStreamState::SEEK_DOLLAR:
            if (b == '$') begin_dollar_candidate();
            break;

        case GnssStreamState::AFTER_DOLLAR:
            if (b == '@') {
                dbg_sbf_sync++;
                sbf_buf[1] = b;
                sbf_idx = 2;
                gnss_stream_state = GnssStreamState::SBF_BODY;
            } else {
                // The only SBF prefix is '$@'. Anything else after '$' selects
                // the requested unconditional NMEA passthrough path.
                if (select_input_protocol(GnssInputProtocol::NMEA)) {
                    const uint8_t prefix[2] = {'$', b};
                    forward_nmea_bytes(prefix, sizeof(prefix));
                }
                reset_gnss_stream_parser();
            }
            break;

        case GnssStreamState::SBF_BODY:
            if (sbf_idx >= sizeof(sbf_buf)) {
                dbg_sbf_len_fail++;
                reset_gnss_stream_parser();
                break;
            }
            sbf_buf[sbf_idx++] = b;
            if (sbf_idx == 8) {
                memcpy(&sbf_len, &sbf_buf[6], sizeof(sbf_len));
                if (sbf_len < 8 || (sbf_len % 4) != 0) {
                    dbg_sbf_len_fail++;
                    reset_gnss_stream_parser();
                } else if (sbf_len > sizeof(sbf_buf)) {
                    // Preserve frame boundaries even for valid SBF block types
                    // which are larger than the blocks this bridge decodes.
                    // Scanning their binary payload for '$' could falsely lock
                    // AUTO mode to an embedded NMEA-looking byte sequence.
                    dbg_sbf_oversize_drop++;
                    sbf_drop_remaining = (uint32_t)sbf_len - sbf_idx;
                    gnss_stream_state = GnssStreamState::DROP_SBF_REMAINDER;
                } else if (sbf_len == sbf_idx) {
                    finish_sbf_candidate();
                    reset_gnss_stream_parser();
                }
            } else if (sbf_idx >= 8 && sbf_idx == sbf_len) {
                finish_sbf_candidate();
                reset_gnss_stream_parser();
            }
            break;

        case GnssStreamState::DROP_SBF_REMAINDER:
            if (sbf_drop_remaining > 0) {
                sbf_drop_remaining--;
            }
            if (sbf_drop_remaining == 0) {
                reset_gnss_stream_parser();
            }
            break;
    }
}

// =========================================================================
// Status printer.
// =========================================================================
static uint32_t last_status_ms = 0;
static uint32_t last_pvt_count = 0, last_pcv_count = 0, last_vcv_count = 0;
static uint32_t last_att_count = 0, last_acv_count = 0;
static uint32_t last_mip_count = 0;
static uint32_t last_nmea_forwarded_bytes = 0;
static uint32_t last_pps_count = 0;
static uint32_t last_cv7_ack_count = 0;
static uint32_t last_cv7_nack_count = 0;
static bool aiding_csv_header_printed = false;

static const char* mode_name(uint8_t m) {
    switch (m) {
        case 0: return "No PVT";
        case 1: return "Stand-Alone";
        case 2: return "DGPS";
        case 3: return "Fixed loc";
        case 4: return "RTK Fixed";
        case 5: return "RTK Float";
        case 6: return "SBAS";
        case 7: return "moving-base RTK Fix";
        case 8: return "moving-base RTK Float";
        case 10: return "PPP";
        default: return "?";
    }
}

static void print_aiding_csv_header_once() {
    if (!PRINT_AIDING_CSV_TO_USB || aiding_csv_header_printed) {
        return;
    }
    Serial.println(F(
        "TEENSY_GPS_AID_HEADER,"
        "teensy_ms,gps_week,tow_ms,mode,num_sats,"
        "lat_deg,lon_deg,height_m,"
        "vel_n_mps,vel_e_mps,vel_d_mps,"
        "pos_sigma_n_m,pos_sigma_e_m,pos_sigma_u_m,"
        "vel_sigma_n_mps,vel_sigma_e_mps,vel_sigma_d_mps,"
        "pos_valid_flags_hex,vel_valid_flags_hex,"
        "heading_valid,heading_tow_ms,heading_deg,heading_sigma_deg,"
        "att_mode,att_error,att_num_sats,att_cov_valid,att_cov_tow_ms,att_cov_error"
    ));
    aiding_csv_header_printed = true;
}

static void print_aiding_csv_row(const mip_time& t, uint16_t pos_flags, uint16_t vel_flags) {
    if (!PRINT_AIDING_CSV_TO_USB) {
        return;
    }
    print_aiding_csv_header_once();
    Serial.print(F("TEENSY_GPS_AID,"));
    Serial.print(millis());
    Serial.print(',');
    Serial.print(pvt.week);
    Serial.print(',');
    Serial.print(pvt.tow_ms);
    Serial.print(',');
    Serial.print(pvt.mode);
    Serial.print(',');
    Serial.print(pvt.num_sats);
    Serial.print(',');
    Serial.print(pvt.lat_deg, 10);
    Serial.print(',');
    Serial.print(pvt.lon_deg, 10);
    Serial.print(',');
    Serial.print(pvt.h_m, 4);
    Serial.print(',');
    Serial.print(pvt.vn, 6);
    Serial.print(',');
    Serial.print(pvt.ve, 6);
    Serial.print(',');
    Serial.print(-pvt.vu, 6);
    Serial.print(',');
    Serial.print(pcv.std_n, 4);
    Serial.print(',');
    Serial.print(pcv.std_e, 4);
    Serial.print(',');
    Serial.print(pcv.std_u, 4);
    Serial.print(',');
    Serial.print(vcv.std_vn, 4);
    Serial.print(',');
    Serial.print(vcv.std_ve, 4);
    Serial.print(',');
    Serial.print(vcv.std_vu, 4);
    Serial.print(F(",0x"));
    Serial.print(pos_flags, HEX);
    Serial.print(F(",0x"));
    Serial.print(vel_flags, HEX);
    Serial.print(',');
    Serial.print(att.valid ? 1 : 0);
    Serial.print(',');
    Serial.print(att.tow_ms);
    Serial.print(',');
    Serial.print(att.heading_deg, 6);
    Serial.print(',');
    Serial.print(acv.valid ? acv.std_heading : 0.0f, 6);
    Serial.print(',');
    Serial.print(att.mode);
    Serial.print(',');
    Serial.print(att.err);
    Serial.print(',');
    Serial.print(att.nr_sv);
    Serial.print(',');
    Serial.print(acv.valid ? 1 : 0);
    Serial.print(',');
    Serial.print(acv.tow_ms);
    Serial.print(',');
    Serial.print(acv.err);
    Serial.println();
    (void)t;
}

static void print_status() {
    noInterrupts();
    const uint32_t pps_snapshot = pps_count;
    const uint32_t last_pps_us_snapshot = last_pps_us;
    interrupts();

    Serial.println();
    Serial.println(F("================ [STATUS] ================"));
    Serial.print(F(" uptime          : ")); Serial.print(millis()/1000); Serial.println(F(" s"));
    Serial.print(F(" GNSS input      : "));
    Serial.println(input_protocol_name(active_input_protocol));
    Serial.println(F(" CV7 settings    : managed externally by GUI"));
    Serial.print(F(" PPS rate        : "));
    Serial.print(pps_snapshot - last_pps_count);
    Serial.print(F(" /s   total: "));
    Serial.print(pps_snapshot);
    Serial.print(F("   last_us: "));
    Serial.println(last_pps_us_snapshot);
    Serial.print(F(" CV7 Time Sync   : "));
    if (!cv7_time_sync_status_received) {
        Serial.println(F("no status received"));
    } else {
        Serial.print(cv7_time_sync_ready() ? F("VALID") : F("INVALID/STALE"));
        Serial.print(F("  last_pps_rcvd="));
        Serial.print(cv7_time_sync_status.last_pps_rcvd);
        Serial.print(F(" s  status_age="));
        Serial.print((uint32_t)(millis() - cv7_time_sync_status_ms));
        Serial.println(F(" ms"));
    }
    Serial.print(F(" GPS Time Update : "));
    if (last_gps_time_update_week == INVALID_GNSS_WEEK) {
        Serial.println(F("none"));
    } else {
        Serial.print(F("week=")); Serial.print(last_gps_time_update_week);
        Serial.print(F(" tow_s=")); Serial.print(last_gps_time_update_tow_s);
        Serial.print(F(" PPS_delay_us=")); Serial.println(last_gps_time_update_delay_us);
    }
    Serial.print(F(" SBF rates       : "));
    Serial.print(F("PVT="));    Serial.print(cnt_pvt    - last_pvt_count);
    Serial.print(F(" PosCov="));Serial.print(cnt_poscov - last_pcv_count);
    Serial.print(F(" VelCov="));Serial.print(cnt_velcov - last_vcv_count);
    Serial.print(F(" Att="));   Serial.print(cnt_att    - last_att_count);
    Serial.print(F(" AttCov="));Serial.print(cnt_attcov - last_acv_count);
    Serial.println(F(" /s"));
    Serial.print(F(" MIP rate        : "));
    Serial.print(mip_total - last_mip_count); Serial.print(F(" /s   total: "));
    Serial.print(mip_total); Serial.print(F("   skipped: "));
    Serial.println(mip_skipped);
    Serial.print(F(" NMEA pass bytes : "));
    Serial.print(nmea_forwarded_bytes - last_nmea_forwarded_bytes);
    Serial.print(F(" /s   total: "));
    Serial.println(nmea_forwarded_bytes);
    Serial.print(F(" CV7 ACK rate    : "));
    Serial.print(cv7_ack_total - last_cv7_ack_count);
    Serial.print(F(" /s   total: "));
    Serial.print(cv7_ack_total);
    Serial.print(F("   NACK total: "));
    Serial.println(cv7_nack_total);
    if (cnt_pvt > 0) {
        Serial.println(F(" ---- LATEST PVT ----"));
        Serial.print(F(" GPS time        : Week "));
        Serial.print(pvt.week); Serial.print(F("  TOW "));
        Serial.print(pvt.tow_ms / 1000.0, 3); Serial.println(F(" s"));
        Serial.print(F(" Mode            : "));
        Serial.print(pvt.mode); Serial.print(F(" ("));
        Serial.print(mode_name(pvt.mode)); Serial.print(F(")  raw=0x"));
        Serial.print(pvt.mode_raw, HEX);
        Serial.print(F("  2D=")); Serial.print(pvt.is_2d ? 1 : 0);
        Serial.print(F("  err=")); Serial.print(pvt.err);
        Serial.print(F("  datum=")); Serial.println(pvt.datum);
        Serial.print(F(" Lat / Lon / H   : "));
        Serial.print(pvt.lat_deg, 8); Serial.print(F("  "));
        Serial.print(pvt.lon_deg, 8); Serial.print(F("  "));
        Serial.print(pvt.h_m, 3);     Serial.println(F(" m"));
        if (pcv.valid) {
            Serial.print(F(" sigma N/E/U     : "));
            Serial.print(pcv.std_n, 3); Serial.print(F(" "));
            Serial.print(pcv.std_e, 3); Serial.print(F(" "));
            Serial.print(pcv.std_u, 3); Serial.println(F(" m"));
        }
        Serial.print(F(" Sat used        : ")); Serial.println(pvt.num_sats);
    }
    if (cnt_att > 0) {
        Serial.print(F(" AttEuler        : TOW="));
        Serial.print(att.tow_ms / 1000.0, 3);
        Serial.print(F(" mode="));
        Serial.print(att.mode);
        Serial.print(F(" err="));
        Serial.print(att.err);
        Serial.print(F(" sv="));
        Serial.print(att.nr_sv);
        Serial.print(F(" valid="));
        Serial.println(att.valid ? F("yes") : F("no"));
        Serial.print(F(" Heading         : "));
        Serial.print(att.heading_deg, 2); Serial.print(F(" deg"));
        if (acv.valid) {
            Serial.print(F("  sigma="));
            Serial.print(acv.std_heading, 2); Serial.print(F(" deg"));
        }
        Serial.print(F("  AttCov TOW="));
        Serial.print(acv.tow_ms / 1000.0, 3);
        Serial.print(F(" err="));
        Serial.print(acv.err);
        Serial.print(F(" valid="));
        Serial.print(acv.valid ? F("yes") : F("no"));
        Serial.println();
    }
    if (DEBUG_MODE) {
        Serial.println(F(" ---- DEBUG ----"));
        Serial.print(F(" Serial1 bytes   : ")); Serial.println(dbg_serial1_bytes);
        Serial.print(F(" SBF sync        : ")); Serial.println(dbg_sbf_sync);
        Serial.print(F(" SBF len fail    : ")); Serial.println(dbg_sbf_len_fail);
        Serial.print(F(" SBF oversize drop: ")); Serial.println(dbg_sbf_oversize_drop);
        Serial.print(F(" SBF CRC ok/fail : "));
        Serial.print(dbg_sbf_crc_ok); Serial.print(F(" / ")); Serial.println(dbg_sbf_crc_fail);
        Serial.print(F(" Unknown block   : ")); Serial.println(dbg_unknown_block);
        Serial.print(F(" frame timeout   : ")); Serial.println(dbg_gnss_frame_timeout);
        Serial.print(F(" protocol conflict: ")); Serial.println(dbg_protocol_conflict);
        Serial.print(F(" skip no PVT blk : ")); Serial.println(dbg_skip_no_pvt);
        Serial.print(F(" skip no PCV blk : ")); Serial.println(dbg_skip_no_poscov);
        Serial.print(F(" skip no VCV blk : ")); Serial.println(dbg_skip_no_velcov);
        Serial.print(F(" skip TOW mismatch: ")); Serial.println(dbg_skip_tow_mismatch);
        Serial.print(F(" skip week mismatch: ")); Serial.println(dbg_skip_week_mismatch);
        Serial.print(F(" skip time-sync   : ")); Serial.println(dbg_skip_time_not_synced);
        Serial.print(F(" skip non-WGS84   : ")); Serial.println(dbg_skip_non_wgs84);
        Serial.print(F(" invalid mode out: ")); Serial.println(dbg_invalid_mode_output);
        Serial.print(F(" skip repeat TOW : ")); Serial.println(dbg_skip_repeat_tow);
        Serial.print(F(" serializer fail : ")); Serial.println(dbg_serializer_fail);
        Serial.print(F(" Serial2 fail    : ")); Serial.println(dbg_serial2_write_fail);
        Serial.print(F(" CV7 ACK POS/VEL : "));
        Serial.print(cv7_ack_pos_llh); Serial.print(F(" / "));
        Serial.println(cv7_ack_vel_ned);
        Serial.print(F(" CV7 ACK TIME/HDG: "));
        Serial.print(cv7_ack_gps_time); Serial.print(F(" / "));
        Serial.println(cv7_ack_heading);
        Serial.print(F(" Last CV7 ACK    : ds=0x"));
        Serial.print(last_cv7_ack_set, HEX);
        Serial.print(F(" field=0x"));
        Serial.print(last_cv7_ack_field, HEX);
        Serial.print(F(" result=0x"));
        Serial.print(last_cv7_ack_result, HEX);
        Serial.print(F(" "));
        Serial.println(ack_result_name(last_cv7_ack_result));
        Serial.print(F(" valid PVT/PCV/VCV: "));
        Serial.print(pvt.valid); Serial.print(F(" / "));
        Serial.print(pcv.valid); Serial.print(F(" / "));
        Serial.println(vcv.valid);
        Serial.print(F(" Aiding flags P/V: 0x"));
        Serial.print(last_pos_valid_flags, HEX); Serial.print(F(" / 0x"));
        Serial.println(last_vel_valid_flags, HEX);
        Serial.print(F(" TOW PVT/PCV/VCV : "));
        Serial.print(pvt.tow_ms); Serial.print(F(" / "));
        Serial.print(pcv.tow_ms); Serial.print(F(" / "));
        Serial.println(vcv.tow_ms);
        Serial.print(F(" WNc PVT/PCV/VCV : "));
        Serial.print(pvt.week); Serial.print(F(" / "));
        Serial.print(pcv.week); Serial.print(F(" / "));
        Serial.println(vcv.week);
    }
    Serial.println(F("=========================================="));

    last_pvt_count = cnt_pvt;
    last_pcv_count = cnt_poscov;
    last_vcv_count = cnt_velcov;
    last_att_count = cnt_att;
    last_acv_count = cnt_attcov;
    last_mip_count = mip_total;
    last_nmea_forwarded_bytes = nmea_forwarded_bytes;
    last_pps_count = pps_snapshot;
    last_cv7_ack_count = cv7_ack_total;
    last_cv7_nack_count = cv7_nack_total;
}

// =========================================================================
// SETUP / LOOP
// =========================================================================
void setup() {
    Serial.begin(USB_BAUD);
    Serial1.begin(GNSS_BAUD);
    Serial2.begin(IMU_BAUD);
    Serial1.addMemoryForRead(serial1_rx_buf, sizeof(serial1_rx_buf));

    pinMode(PPS_IN_PIN, INPUT);
    pinMode(PPS_OUT_IMU_PIN, OUTPUT);
    pinMode(PPS_LED_PIN, OUTPUT);
    digitalWriteFast(PPS_OUT_IMU_PIN, digitalReadFast(PPS_IN_PIN));
    digitalWriteFast(PPS_LED_PIN, LOW);
    attachInterrupt(digitalPinToInterrupt(PPS_IN_PIN), pps_isr, CHANGE);

    if (PRINT_STATUS_TO_USB) {
        Serial.println();
        Serial.println(F("================================================"));
        Serial.println(F(" Teensy 4.1  SBF/NMEA -> CV7 Bridge (MIP SDK)"));
        Serial.println(F(" Serial1 (Pin 0): mosaic-H SBF or NMEA @ 115200"));
        Serial.println(F(" Serial2 (Pin 8): CV7 UART1 MIP + NMEA @ 115200"));
        Serial.print(F(" PPS in/out     : Pin "));
        Serial.print(PPS_IN_PIN);
        Serial.print(F(" -> Pin "));
        Serial.println(PPS_OUT_IMU_PIN);
        Serial.print(F(" PPS LED        : Pin "));
        Serial.println(PPS_LED_PIN);
        Serial.print(F(" Debug mode     : "));
        Serial.println(DEBUG_MODE ? F("ON (USB text diagnostics)") : F("OFF"));
        Serial.print(F(" Heading aiding : "));
        Serial.println(EXTERNAL_HEADING_FRAME_CONFIGURED
            ? F("ENABLED (vehicle-frame alignment confirmed)")
            : F("DISABLED (alignment not confirmed)"));
        Serial.println(F(" Aiding time    : EXTERNAL_TIME (GPS week/TOW via PPS)"));
        Serial.print(F(" Input selector : "));
        Serial.println(input_protocol_name(GNSS_INPUT_PROTOCOL));
        Serial.println(F("================================================"));
    }

    // Initialize MIP SDK interface
    mip_interface_init(&mip_dev,
                       100,            // parser timeout (ms)
                       2000,           // command reply timeout (ms)
                       &app_mip_send_callback,
                       &app_mip_recv_callback,
                       &app_mip_update_callback,
                       NULL);

    mip_interface_register_field_callback(
        &mip_dev,
        &cv7_time_sync_handler,
        MIP_SYSTEM_DATA_DESC_SET,
        MIP_DATA_DESC_SYSTEM_TIME_SYNC_STATUS,
        &handle_cv7_time_sync_status,
        NULL);

    if (PRINT_STATUS_TO_USB) {
        Serial.println(F("[init] CV7 settings are managed externally by GUI."));
        Serial.println(F("[init] This firmware sends no CV7 configuration commands."));
        Serial.println(F("[init] Waiting for SBF '$@' or an NMEA '$x' prefix."));
        Serial.println();
    }
    last_status_ms = millis();
}

void loop() {
    while (Serial1.available()) {
        int b = Serial1.read();
        if (b < 0) break;
        feed_serial1_byte((uint8_t)b);
    }

    // Pump MIP SDK (handles any pending replies from CV7)
    mip_interface_update(&mip_dev, 0, false);

    service_pps_led();

    if (PRINT_STATUS_TO_USB && millis() - last_status_ms >= 1000) {
        last_status_ms = millis();
        print_status();
    }
}
