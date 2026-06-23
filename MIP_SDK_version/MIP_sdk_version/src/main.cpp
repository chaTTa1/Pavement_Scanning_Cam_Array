/*
 * Teensy 4.1  SBF -> MIP Bridge  (using official MicroStrain MIP SDK)
 * ====================================================================
 * Hardware:
 *   Septentrio mosaic-H SBF UART TX  --> Teensy Pin 0 (Serial1 RX)
 *   Septentrio mosaic-H PPS output   --> Teensy Pin 2 (PPS_IN_PIN)
 *   Common GND                       --- Teensy GND / CV7 GND
 *
 *   Teensy Pin 8 (Serial2 TX) --> CV7 C-Series UUT_RX / RxD
 *   Teensy Pin 7 (Serial2 RX) <-- CV7 C-Series UUT_TX / TxD
 *   Teensy Pin 3 (PPS_OUT_IMU_PIN) --> CV7 GPIO1 configured as PPS input
 *   CV7 power/readout via C-Series USB.
 *
 * MIP descriptor set: 0x13 (Aiding)
 *   0x01  Aiding Frame Configuration       (one-shot in setup)
 *   0x22  External Position LLH            (10 Hz from GNSS)
 *   0x29  External Velocity NED            (10 Hz from GNSS)
 *   0x31  External Heading True            (10 Hz from mosaic-H AttEuler, if valid)
 * Filter command set: 0x0D
 *   0x50  Aiding Measurement Enable        (one-shot in setup)
 */

#include <Arduino.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "mip/mip_all.h"
#include "mip/definitions/commands_aiding.h"
#include "mip/definitions/commands_base.h"

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

// GNSS antenna lever arm in IMU body frame [m]
// Update with real measurement when mounting on vehicle
static constexpr float ANTENNA_LEVER_ARM[3] = {0.0f, 0.0f, 0.0f};

// CV7/GV7 SDK mip_time format used by Aiding commands:
//   timebase u8, reserved u8 (=1), nanoseconds u64.
// EXTERNAL_TIME tells the CV7 to interpret aiding timestamps against the
// external GPS/PPS timebase configured with GPS Time Update (0x01,0x72).
static constexpr mip_time_timebase AIDING_TIMEBASE = MIP_TIME_TIMEBASE_EXTERNAL_TIME;
static constexpr bool USE_GPS_TIME_NANOSECONDS = true;
static constexpr uint32_t INVALID_GNSS_TOW_MS = 0xFFFFFFFFu;
static constexpr uint32_t HEADING_MAX_TOW_DELTA_MS = 200u;

// Keep external GNSS velocity aiding enabled, but mirror the exact aiding values
// sent to the CV7 as CSV over USB debug output for audit/comparison with NMEA.
static constexpr bool SEND_VELOCITY_AIDING = true;
static constexpr bool PRINT_AIDING_CSV_TO_USB = DEBUG_MODE;

// CV7 aiding frame ID (we define this; any value 1..255 works)
static constexpr uint8_t AIDING_FRAME_ID = 1;

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

static uint32_t tow_abs_delta_ms(uint32_t a, uint32_t b) {
    return (a > b) ? (a - b) : (b - a);
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
    uint8_t  mode = 0;
    uint8_t  err = 0;
    double   lat_deg = 0, lon_deg = 0, h_m = 0;
    float    vn = 0, ve = 0, vu = 0;
    uint8_t  num_sats = 0;
};
struct PosCovSnap {
    bool valid = false; uint32_t tow_ms = 0;
    float std_n = 0, std_e = 0, std_u = 0;
};
struct VelCovSnap {
    bool valid = false; uint32_t tow_ms = 0;
    float std_vn = 0, std_ve = 0, std_vu = 0;
};
struct AttSnap {
    bool valid = false; uint32_t tow_ms = 0;
    uint8_t nr_sv = 0, err = 0;
    uint16_t mode = 0;
    float heading_deg_no_reserved = 0;
    float heading_deg = 0, pitch_deg = 0, roll_deg = 0;
};
struct AttCovSnap {
    bool valid = false;
    uint32_t tow_ms = 0;
    uint8_t err = 0;
    float std_heading = 0, std_pitch = 0, std_roll = 0;
};

static PvtSnapshot pvt;
static PosCovSnap  pcv;
static VelCovSnap  vcv;
static AttSnap     att;
static AttCovSnap  acv;

static uint32_t cnt_pvt = 0, cnt_poscov = 0, cnt_velcov = 0, cnt_att = 0, cnt_attcov = 0;
static uint32_t mip_total = 0, mip_skipped = 0;
static uint32_t last_mip_tow = INVALID_GNSS_TOW_MS;
static uint32_t last_heading_mip_tow = INVALID_GNSS_TOW_MS;
static uint32_t last_mip_velcov_count = 0;
static uint16_t last_gps_time_update_week = 0xFFFFu;
static uint32_t last_gps_time_update_tow_s = 0xFFFFFFFFu;
static uint16_t last_pos_valid_flags = 0;
static uint16_t last_vel_valid_flags = 0;

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

static uint32_t dbg_sbf_bytes = 0;
static uint32_t dbg_sbf_sync = 0;
static uint32_t dbg_sbf_len_fail = 0;
static uint32_t dbg_sbf_crc_ok = 0;
static uint32_t dbg_sbf_crc_fail = 0;
static uint32_t dbg_unknown_block = 0;

static uint32_t dbg_skip_no_pvt = 0;
static uint32_t dbg_skip_no_poscov = 0;
static uint32_t dbg_skip_no_velcov = 0;
static uint32_t dbg_skip_tow_mismatch = 0;
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

static void parse_pvt_geodetic(const uint8_t* blk, uint16_t) {
    PvtSnapshot s;
    memcpy(&s.tow_ms, blk + 8, 4);
    memcpy(&s.week,   blk + 12, 2);
    s.mode = blk[14] & 0x0F;
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
    s.num_sats = blk[74];
    s.valid = !(is_dnu_d(s.lat_deg) || is_dnu_d(s.lon_deg) || is_dnu_d(s.h_m)
             || is_dnu_f(s.vn) || is_dnu_f(s.ve) || is_dnu_f(s.vu));
    pvt = s;
    cnt_pvt++;
}

static void parse_pos_cov_geodetic(const uint8_t* blk, uint16_t) {
    PosCovSnap s;
    memcpy(&s.tow_ms, blk + 8, 4);
    float var_n, var_e, var_u;
    memcpy(&var_n, blk + 16, 4);
    memcpy(&var_e, blk + 20, 4);
    memcpy(&var_u, blk + 24, 4);
    s.std_n = (var_n > 0 && !is_dnu_f(var_n)) ? sqrtf(var_n) : 0.0f;
    s.std_e = (var_e > 0 && !is_dnu_f(var_e)) ? sqrtf(var_e) : 0.0f;
    s.std_u = (var_u > 0 && !is_dnu_f(var_u)) ? sqrtf(var_u) : 0.0f;
    s.valid = !(is_dnu_f(var_n) || is_dnu_f(var_e) || is_dnu_f(var_u));
    pcv = s;
    cnt_poscov++;
}

static void parse_vel_cov_geodetic(const uint8_t* blk, uint16_t) {
    VelCovSnap s;
    memcpy(&s.tow_ms, blk + 8, 4);
    float var_vn, var_ve, var_vu;
    memcpy(&var_vn, blk + 16, 4);
    memcpy(&var_ve, blk + 20, 4);
    memcpy(&var_vu, blk + 24, 4);
    s.std_vn = (var_vn > 0 && !is_dnu_f(var_vn)) ? sqrtf(var_vn) : 0.0f;
    s.std_ve = (var_ve > 0 && !is_dnu_f(var_ve)) ? sqrtf(var_ve) : 0.0f;
    s.std_vu = (var_vu > 0 && !is_dnu_f(var_vu)) ? sqrtf(var_vu) : 0.0f;
    s.valid = !(is_dnu_f(var_vn) || is_dnu_f(var_ve) || is_dnu_f(var_vu));
    vcv = s;
    cnt_velcov++;
}

static void parse_att_euler(const uint8_t* blk, uint16_t) {
    AttSnap s;
    memcpy(&s.tow_ms, blk + 8, 4);
    s.nr_sv = blk[14];
    s.err = blk[15];
    memcpy(&s.mode, blk + 16, 2);
    memcpy(&s.heading_deg_no_reserved, blk + 18, 4);
    memcpy(&s.heading_deg, blk + 20, 4);
    memcpy(&s.pitch_deg,   blk + 24, 4);
    memcpy(&s.roll_deg,    blk + 28, 4);
    s.valid = (s.err == 0) && (s.mode >= 1 && s.mode <= 4) && !is_dnu_f(s.heading_deg);
    att = s;
    cnt_att++;
}

static void parse_att_cov_euler(const uint8_t* blk, uint16_t) {
    AttCovSnap s;
    memcpy(&s.tow_ms, blk + 8, 4);
    s.err = blk[15];
    float var_h, var_p, var_r;
    memcpy(&var_h, blk + 16, 4);
    memcpy(&var_p, blk + 20, 4);
    memcpy(&var_r, blk + 24, 4);
    s.std_heading = (var_h > 0 && !is_dnu_f(var_h)) ? sqrtf(var_h) : 0.0f;
    s.std_pitch   = (var_p > 0 && !is_dnu_f(var_p)) ? sqrtf(var_p) : 0.0f;
    s.std_roll    = (var_r > 0 && !is_dnu_f(var_r)) ? sqrtf(var_r) : 0.0f;
    s.valid = (s.err == 0) && !is_dnu_f(var_h);
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

static bool send_gps_time_update_field(mip_base_gps_time_update_command_field_id field_id,
                                       uint32_t value) {
    microstrain_serializer s;
    uint8_t buf[MIP_FIELD_PAYLOAD_LENGTH_MAX];
    microstrain_serializer_init_insertion(&s, buf, sizeof(buf));

    insert_mip_function_selector(&s, MIP_FUNCTION_WRITE);
    insert_mip_base_gps_time_update_command_field_id(&s, field_id);
    microstrain_insert_u32(&s, value);

    if (!microstrain_serializer_is_ok(&s)) {
        return false;
    }

    return mip_send_command_field(MIP_CMD_DESC_BASE_GPS_TIME_UPDATE,
                                  buf,
                                  microstrain_serializer_length(&s),
                                  MIP_BASE_CMD_DESC_SET);
}

static void maybe_send_gps_time_update(uint16_t week, uint32_t tow_ms) {
    const uint32_t tow_s = tow_ms / 1000u;
    if (week == last_gps_time_update_week && tow_s == last_gps_time_update_tow_s) {
        return;
    }

    const bool sent_tow = send_gps_time_update_field(
        MIP_BASE_GPS_TIME_UPDATE_COMMAND_FIELD_ID_TIME_OF_WEEK, tow_s);
    const bool sent_week = send_gps_time_update_field(
        MIP_BASE_GPS_TIME_UPDATE_COMMAND_FIELD_ID_WEEK_NUMBER, week);

    if (sent_tow && sent_week) {
        last_gps_time_update_week = week;
        last_gps_time_update_tow_s = tow_s;
    }
}

// =========================================================================
// MIP SETUP COMMANDS (sent once in setup())
// =========================================================================

// 1. Aiding Frame Configuration (0x13, 0x01)
//    Tells CV7 where the GNSS antenna is relative to the IMU
static bool send_frame_config() {
    mip_aiding_frame_config_command cmd;
    memset(&cmd, 0, sizeof(cmd));

    cmd.function = MIP_FUNCTION_WRITE;
    cmd.frame_id = AIDING_FRAME_ID;
    cmd.format   = MIP_AIDING_FRAME_CONFIG_COMMAND_FORMAT_EULER;
    cmd.tracking_enabled = false;
    cmd.translation[0] = ANTENNA_LEVER_ARM[0];
    cmd.translation[1] = ANTENNA_LEVER_ARM[1];
    cmd.translation[2] = ANTENNA_LEVER_ARM[2];
    cmd.rotation.euler[0] = 0.0f;
    cmd.rotation.euler[1] = 0.0f;
    cmd.rotation.euler[2] = 0.0f;

    mip_cmd_result r = mip_aiding_write_frame_config(&mip_dev,
        cmd.frame_id, cmd.format, cmd.tracking_enabled, cmd.translation,
        &cmd.rotation);
    return mip_cmd_result_is_ack(r);
}

// 2. Aiding Measurement Enable (filter command set 0x0D, field 0x50)
//    Enables GNSS position/velocity and external heading aiding sources.
static bool send_aiding_enable_all() {
    mip_cmd_result r1 = mip_filter_write_aiding_measurement_enable(&mip_dev,
        MIP_FILTER_AIDING_MEASUREMENT_ENABLE_COMMAND_AIDING_SOURCE_GNSS_POS_VEL,
        true);
    mip_cmd_result r2 = mip_filter_write_aiding_measurement_enable(&mip_dev,
        MIP_FILTER_AIDING_MEASUREMENT_ENABLE_COMMAND_AIDING_SOURCE_EXTERNAL_HEADING,
        true);
    return mip_cmd_result_is_ack(r1) && mip_cmd_result_is_ack(r2);
}

// 3. Save Settings to NVRAM
static bool save_settings_to_startup() {
    mip_cmd_result r = mip_3dm_save_device_settings(&mip_dev);
    return mip_cmd_result_is_ack(r);
}

static bool pvt_mode_allows_valid_aiding() {
    return pvt.mode != 0 && pvt.mode != 3;
}

static uint16_t position_valid_flags() {
    if (!pvt_mode_allows_valid_aiding() || !pcv.valid) {
        return MIP_AIDING_POS_LLH_COMMAND_VALID_FLAGS_NONE;
    }

    uint16_t flags = MIP_AIDING_POS_LLH_COMMAND_VALID_FLAGS_NONE;
    if (!is_dnu_d(pvt.lat_deg)) flags |= MIP_AIDING_POS_LLH_COMMAND_VALID_FLAGS_LATITUDE;
    if (!is_dnu_d(pvt.lon_deg)) flags |= MIP_AIDING_POS_LLH_COMMAND_VALID_FLAGS_LONGITUDE;
    if (!is_dnu_d(pvt.h_m)) flags |= MIP_AIDING_POS_LLH_COMMAND_VALID_FLAGS_HEIGHT;
    return flags;
}

static uint16_t velocity_valid_flags() {
    if (!pvt_mode_allows_valid_aiding() || !vcv.valid) {
        return MIP_AIDING_VEL_NED_COMMAND_VALID_FLAGS_NONE;
    }

    uint16_t flags = MIP_AIDING_VEL_NED_COMMAND_VALID_FLAGS_NONE;
    if (!is_dnu_f(pvt.vn)) flags |= MIP_AIDING_VEL_NED_COMMAND_VALID_FLAGS_X;
    if (!is_dnu_f(pvt.ve)) flags |= MIP_AIDING_VEL_NED_COMMAND_VALID_FLAGS_Y;
    if (!is_dnu_f(pvt.vu)) flags |= MIP_AIDING_VEL_NED_COMMAND_VALID_FLAGS_Z;
    return flags;
}

// =========================================================================
// MIP REALTIME OUTPUT (3 fields per GNSS epoch)
// =========================================================================
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

    // VelCov is the last required block in the configured SBF order. It also
    // gives a 10 Hz send cadence when the receiver reports invalid TOW values.
    if (cnt_velcov == last_mip_velcov_count) {
        return;
    }
    last_mip_velcov_count = cnt_velcov;

    const bool tow_is_valid = pvt.tow_ms != INVALID_GNSS_TOW_MS
                           && pcv.tow_ms != INVALID_GNSS_TOW_MS
                           && vcv.tow_ms != INVALID_GNSS_TOW_MS;
    if (USE_GPS_TIME_NANOSECONDS && !tow_is_valid) {
        mip_skipped++;
        return;
    }
    if (tow_is_valid && (pvt.tow_ms != pcv.tow_ms || pvt.tow_ms != vcv.tow_ms)) {
        dbg_skip_tow_mismatch++;
        return;
    }
    if (tow_is_valid && pvt.tow_ms == last_mip_tow) {
        dbg_skip_repeat_tow++;
        return;
    }
    if (tow_is_valid) {
        last_mip_tow = pvt.tow_ms;
    }
    if (!pvt_mode_allows_valid_aiding()) {
        dbg_invalid_mode_output++;
    }
    if (USE_GPS_TIME_NANOSECONDS) {
        maybe_send_gps_time_update(pvt.week, pvt.tow_ms);
    }

    // Build mip_time once, reuse for all 3 fields
    mip_time t;
    t.timebase = AIDING_TIMEBASE;
    t.reserved = 1;
    t.nanoseconds = USE_GPS_TIME_NANOSECONDS ? gps_time_ns(pvt.week, pvt.tow_ms) : 0ULL;

    // --- Field 1: External Position LLH (0x22) ---
    {
        double position[3] = { pvt.lat_deg, pvt.lon_deg, pvt.h_m };
        float  uncertainty[3] = { pcv.std_n, pcv.std_e, pcv.std_u };

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
        cmd.valid_flags = position_valid_flags();
        last_pos_valid_flags = cmd.valid_flags;

        // Send as one-shot field (no response expected)
        mip_serializer s;
        uint8_t buf[80];
        microstrain_serializer_init_insertion(&s, buf, sizeof(buf));
        insert_mip_aiding_pos_llh_command(&s, &cmd);
        if (microstrain_serializer_is_ok(&s)) {
            if (mip_send_command_field(MIP_CMD_DESC_AIDING_POS_LLH,
                                       buf, microstrain_serializer_length(&s))) {
                mip_total++;
            }
        } else {
            dbg_serializer_fail++;
        }
    }

    // --- Field 2: External Velocity NED (0x29) ---
    if (SEND_VELOCITY_AIDING) {
        mip_aiding_vel_ned_command cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.time = t;
        cmd.frame_id = AIDING_FRAME_ID;
        cmd.velocity[0] = pvt.vn;     // North
        cmd.velocity[1] = pvt.ve;     // East
        cmd.velocity[2] = -pvt.vu;    // Down = -Up
        cmd.uncertainty[0] = vcv.std_vn;
        cmd.uncertainty[1] = vcv.std_ve;
        cmd.uncertainty[2] = vcv.std_vu;
        cmd.valid_flags = velocity_valid_flags();
        last_vel_valid_flags = cmd.valid_flags;

        mip_serializer s;
        uint8_t buf[80];
        microstrain_serializer_init_insertion(&s, buf, sizeof(buf));
        insert_mip_aiding_vel_ned_command(&s, &cmd);
        if (microstrain_serializer_is_ok(&s)) {
            if (mip_send_command_field(MIP_CMD_DESC_AIDING_VEL_NED,
                                       buf, microstrain_serializer_length(&s))) {
                mip_total++;
            }
        } else {
            dbg_serializer_fail++;
        }
    }

    print_aiding_csv_row(t, last_pos_valid_flags, last_vel_valid_flags);

    // --- Field 3: External Heading True (0x31) ---
    //     Sent when dual-antenna attitude is valid and close to this GNSS epoch.
    const bool heading_tow_valid = att.tow_ms != INVALID_GNSS_TOW_MS && pvt.tow_ms != INVALID_GNSS_TOW_MS;
    const bool heading_tow_close = heading_tow_valid
                                && tow_abs_delta_ms(att.tow_ms, pvt.tow_ms) <= HEADING_MAX_TOW_DELTA_MS;
    const bool heading_cov_tow_close = acv.tow_ms != INVALID_GNSS_TOW_MS
                                    && att.tow_ms != INVALID_GNSS_TOW_MS
                                    && tow_abs_delta_ms(acv.tow_ms, att.tow_ms) <= HEADING_MAX_TOW_DELTA_MS;
    if (att.valid && acv.valid && acv.std_heading > 0.0f && heading_tow_close && heading_cov_tow_close
            && att.tow_ms != last_heading_mip_tow) {
        mip_time heading_time = t;
        heading_time.nanoseconds = USE_GPS_TIME_NANOSECONDS ? gps_time_ns(pvt.week, att.tow_ms) : 0ULL;

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
                last_heading_mip_tow = att.tow_ms;
            }
        } else {
            dbg_serializer_fail++;
        }
    }
}

static void handle_sbf_packet(uint16_t blk_id, const uint8_t* blk, uint16_t len) {
    blk_id &= 0x1FFF;
    switch (blk_id) {
        case SBF_PVT_GEODETIC:     parse_pvt_geodetic(blk, len);     break;
        case SBF_POS_COV_GEODETIC: parse_pos_cov_geodetic(blk, len); break;
        case SBF_VEL_COV_GEODETIC: parse_vel_cov_geodetic(blk, len); break;
        case SBF_ATT_EULER:        parse_att_euler(blk, len);        break;
        case SBF_ATT_COV_EULER:    parse_att_cov_euler(blk, len);    break;
        default:
            dbg_unknown_block++;
            return;
    }
    try_send_mip();
}

// =========================================================================
// SBF stream state machine.
// =========================================================================
static uint8_t  sbf_buf[1024];
static uint16_t sbf_idx = 0;
static enum { S_SYNC1, S_SYNC2, S_BODY } sbf_state = S_SYNC1;
static uint16_t sbf_len = 0;

static void feed_serial1_byte(uint8_t b) {
    dbg_sbf_bytes++;

    switch (sbf_state) {
        case S_SYNC1:
            if (b == 0x24) { sbf_buf[0] = b; sbf_idx = 1; sbf_state = S_SYNC2; }
            break;
        case S_SYNC2:
            if (b == 0x40) {
                dbg_sbf_sync++;
                sbf_buf[1] = b;
                sbf_idx = 2;
                sbf_state = S_BODY;
                sbf_len = 0;
            }
            else if (b == 0x24) { sbf_buf[0] = b; sbf_idx = 1; }
            else { sbf_state = S_SYNC1; }
            break;
        case S_BODY:
            if (sbf_idx < sizeof(sbf_buf)) sbf_buf[sbf_idx++] = b;
            else { dbg_sbf_len_fail++; sbf_state = S_SYNC1; sbf_idx = 0; break; }
            if (sbf_idx == 8) {
                memcpy(&sbf_len, &sbf_buf[6], 2);
                if (sbf_len < 8 || sbf_len > sizeof(sbf_buf) || (sbf_len % 4) != 0) {
                    dbg_sbf_len_fail++;
                    sbf_state = S_SYNC1; sbf_idx = 0;
                }
            } else if (sbf_idx >= 8 && sbf_idx == sbf_len) {
                uint16_t crc_expect;
                memcpy(&crc_expect, &sbf_buf[2], 2);
                uint16_t crc_calc = sbf_crc16(&sbf_buf[4], sbf_len - 4);
                if (crc_calc == crc_expect) {
                    dbg_sbf_crc_ok++;
                    uint16_t blk_id;
                    memcpy(&blk_id, &sbf_buf[4], 2);
                    handle_sbf_packet(blk_id, sbf_buf, sbf_len);
                } else {
                    dbg_sbf_crc_fail++;
                }
                sbf_state = S_SYNC1; sbf_idx = 0;
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
    Serial.print(F(" PPS rate        : "));
    Serial.print(pps_snapshot - last_pps_count);
    Serial.print(F(" /s   total: "));
    Serial.print(pps_snapshot);
    Serial.print(F("   last_us: "));
    Serial.println(last_pps_us_snapshot);
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
    Serial.print(F(" CV7 ACK rate    : "));
    Serial.print(cv7_ack_total - last_cv7_ack_count);
    Serial.print(F(" /s   total: "));
    Serial.print(cv7_ack_total);
    Serial.print(F("   NACK total: "));
    Serial.println(cv7_nack_total);
    if (pvt.valid) {
        Serial.println(F(" ---- LATEST PVT ----"));
        Serial.print(F(" GPS time        : Week "));
        Serial.print(pvt.week); Serial.print(F("  TOW "));
        Serial.print(pvt.tow_ms / 1000.0, 3); Serial.println(F(" s"));
        Serial.print(F(" Mode            : "));
        Serial.print(pvt.mode); Serial.print(F(" ("));
        Serial.print(mode_name(pvt.mode)); Serial.println(F(")"));
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
        Serial.print(F("  raw18="));
        Serial.print(att.heading_deg_no_reserved, 2); Serial.print(F(" deg"));
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
        Serial.print(F(" SBF bytes       : ")); Serial.println(dbg_sbf_bytes);
        Serial.print(F(" SBF sync        : ")); Serial.println(dbg_sbf_sync);
        Serial.print(F(" SBF len fail    : ")); Serial.println(dbg_sbf_len_fail);
        Serial.print(F(" SBF CRC ok/fail : "));
        Serial.print(dbg_sbf_crc_ok); Serial.print(F(" / ")); Serial.println(dbg_sbf_crc_fail);
        Serial.print(F(" Unknown block   : ")); Serial.println(dbg_unknown_block);
        Serial.print(F(" skip no PVT blk : ")); Serial.println(dbg_skip_no_pvt);
        Serial.print(F(" skip no PCV blk : ")); Serial.println(dbg_skip_no_poscov);
        Serial.print(F(" skip no VCV blk : ")); Serial.println(dbg_skip_no_velcov);
        Serial.print(F(" skip TOW mismatch: ")); Serial.println(dbg_skip_tow_mismatch);
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
    }
    Serial.println(F("=========================================="));

    last_pvt_count = cnt_pvt;
    last_pcv_count = cnt_poscov;
    last_vcv_count = cnt_velcov;
    last_att_count = cnt_att;
    last_acv_count = cnt_attcov;
    last_mip_count = mip_total;
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

    uint32_t t0 = millis();
    while (!Serial && (millis() - t0) < 3000) {}

    if (PRINT_STATUS_TO_USB) {
        Serial.println();
        Serial.println(F("================================================"));
        Serial.println(F(" Teensy 4.1  SBF -> MIP Bridge (with MIP SDK)"));
        Serial.println(F(" Serial1 (Pin 0): mosaic-H SBF @ 115200"));
        Serial.println(F(" Serial2 (Pin 8): CV7-INS MIP    @ 115200"));
        Serial.print(F(" PPS in/out     : Pin "));
        Serial.print(PPS_IN_PIN);
        Serial.print(F(" -> Pin "));
        Serial.println(PPS_OUT_IMU_PIN);
        Serial.print(F(" PPS LED        : Pin "));
        Serial.println(PPS_LED_PIN);
        Serial.print(F(" Debug mode     : "));
        Serial.println(DEBUG_MODE ? F("ON (USB text diagnostics)") : F("OFF"));
        Serial.println(F(" Aiding time    : EXTERNAL_TIME (GPS week/TOW via PPS)"));
        Serial.println(F("================================================"));
    }

    // Initialize MIP SDK interface
    mip_interface_init(&mip_dev,
                       100,            // parser timeout (ms)
                       1000,           // base reply timeout (ms)
                       &app_mip_send_callback,
                       &app_mip_recv_callback,
                       &app_mip_update_callback,
                       NULL);

    // One-time CV7-INS configuration
    delay(1000);  // wait for CV7 to be ready

    if (PRINT_STATUS_TO_USB) Serial.println(F("[init] Sending Frame Config..."));
    if (send_frame_config()) {
        if (PRINT_STATUS_TO_USB) Serial.println(F("[init]   ACK"));
    } else {
        if (PRINT_STATUS_TO_USB) Serial.println(F("[init]   FAILED (CV7 not connected or busy)"));
    }

    if (PRINT_STATUS_TO_USB) Serial.println(F("[init] Sending Aiding Measurement Enable..."));
    if (send_aiding_enable_all()) {
        if (PRINT_STATUS_TO_USB) Serial.println(F("[init]   ACK"));
    } else {
        if (PRINT_STATUS_TO_USB) Serial.println(F("[init]   FAILED"));
    }

    if (PRINT_STATUS_TO_USB) Serial.println(F("[init] Saving Settings to NVRAM..."));
    if (save_settings_to_startup()) {
        if (PRINT_STATUS_TO_USB) Serial.println(F("[init]   ACK"));
    } else {
        if (PRINT_STATUS_TO_USB) Serial.println(F("[init]   FAILED"));
    }

    if (PRINT_STATUS_TO_USB) {
        Serial.println(F("[init] CV7-INS initialization complete"));
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
