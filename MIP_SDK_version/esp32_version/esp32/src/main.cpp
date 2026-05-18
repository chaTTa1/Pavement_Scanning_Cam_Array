/*
 * ESP32  SBF -> MIP Bridge  (using official MicroStrain MIP SDK)
 * ==============================================================
 * Hardware:
 *   GPS/GNSS SBF TX -> ESP32 GPIO25 (Serial2 RX)
 *   GPS/GNSS RX     <- ESP32 GPIO26 (Serial2 TX, optional)
 *
 *   ESP32 GPIO32 (Serial1 TX) -> CV7-INS / IMU RX
 *   ESP32 GPIO33 (Serial1 RX) <- CV7-INS / IMU TX, optional for ACKs
 *   ESP32 GND                 --- GPS and IMU GND
 *
 * MIP descriptor set: 0x13 (Aiding)
 *   0x01  Aiding Frame Configuration       (one-shot in setup)
 *   0x50  Aiding Measurement Enable        (one-shot in setup)
 *   0x22  External Position LLH            (10 Hz from GNSS)
 *   0x29  External Velocity NED            (10 Hz from GNSS)
 *   0x31  External Heading True            (10 Hz from GNSS, if dual-ant)
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

static constexpr int GPS_RX_PIN = 25;  // ESP32 RX, connect to GPS TX
static constexpr int GPS_TX_PIN = 26;  // ESP32 TX, optional connect to GPS RX
static constexpr int IMU_RX_PIN = 33;  // ESP32 RX, optional connect to IMU TX
static constexpr int IMU_TX_PIN = 32;  // ESP32 TX, connect to IMU RX

// Set true when you want to capture the generated MIP stream from the
// ESP32 USB COM port. USB status text is disabled in this mode so the
// capture remains a clean binary MIP file.
static constexpr bool MIRROR_MIP_TO_USB = false;
static constexpr bool PRINT_STATUS_TO_USB = !MIRROR_MIP_TO_USB;

// PPS forwarding:
//   GPS 1 Hz PPS -> PPS_IN_PIN
//   PPS_OUT_IMU_PIN -> CV7 PPS input
// Change these two pins to match your wiring.
static constexpr uint8_t PPS_IN_PIN = 27;
static constexpr uint8_t PPS_OUT_IMU_PIN = 14;

// GNSS antenna lever arm in IMU body frame [m]
// Update with real measurement when mounting on vehicle
static constexpr float ANTENNA_LEVER_ARM[3] = {0.0f, 0.0f, 0.0f};

// CV7/GV7 SDK mip_time format used by Aiding commands:
//   timebase u8, reserved u8 (=1), nanoseconds u64.
// TIME_OF_ARRIVAL is the safest default unless CV7 is synced to GNSS PPS or
// another external clock. If external-time sync is configured, switch to
// EXTERNAL_TIME and enable GPS nanoseconds below.
static constexpr mip_time_timebase AIDING_TIMEBASE = MIP_TIME_TIMEBASE_TIME_OF_ARRIVAL;
static constexpr bool USE_GPS_TIME_NANOSECONDS = false;

// CV7 aiding frame ID (we define this; any value 1..255 works)
static constexpr uint8_t AIDING_FRAME_ID = 1;

static volatile uint32_t pps_count = 0;
static volatile uint32_t last_pps_us = 0;

// =========================================================================
// SBF parser (same as before, unchanged)
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

static void IRAM_ATTR pps_isr() {
    const bool level = digitalRead(PPS_IN_PIN);
    digitalWrite(PPS_OUT_IMU_PIN, level);

    if (level) {
        pps_count++;
        last_pps_us = micros();
    }
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
    float heading_deg = 0, pitch_deg = 0, roll_deg = 0;
};
struct AttCovSnap {
    bool valid = false;
    float std_heading = 0, std_pitch = 0, std_roll = 0;
};

static PvtSnapshot pvt;
static PosCovSnap  pcv;
static VelCovSnap  vcv;
static AttSnap     att;
static AttCovSnap  acv;

static uint32_t cnt_pvt = 0, cnt_poscov = 0, cnt_velcov = 0, cnt_att = 0, cnt_attcov = 0;
static uint32_t mip_total = 0, mip_skipped = 0;
static uint32_t last_mip_tow = 0xFFFFFFFFu;

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
    memcpy(&s.heading_deg, blk + 16, 4);
    memcpy(&s.pitch_deg,   blk + 20, 4);
    memcpy(&s.roll_deg,    blk + 24, 4);
    s.valid = !is_dnu_f(s.heading_deg);
    att = s;
    cnt_att++;
}

static void parse_att_cov_euler(const uint8_t* blk, uint16_t) {
    AttCovSnap s;
    float var_h, var_p, var_r;
    memcpy(&var_h, blk + 16, 4);
    memcpy(&var_p, blk + 20, 4);
    memcpy(&var_r, blk + 24, 4);
    s.std_heading = (var_h > 0 && !is_dnu_f(var_h)) ? sqrtf(var_h) : 0.0f;
    s.std_pitch   = (var_p > 0 && !is_dnu_f(var_p)) ? sqrtf(var_p) : 0.0f;
    s.std_roll    = (var_r > 0 && !is_dnu_f(var_r)) ? sqrtf(var_r) : 0.0f;
    s.valid = !is_dnu_f(var_h);
    acv = s;
    cnt_attcov++;
}

// =========================================================================
// MIP SDK interface adapter
// =========================================================================
static mip_interface mip_dev;

static bool app_mip_send_callback(mip_interface* /*device*/,
                                  const uint8_t* data, size_t length) {
    size_t written = Serial1.write(data, length);
    Serial1.flush();
    return written == length;
}

static bool app_mip_recv_callback(mip_interface* /*device*/,
                                  uint8_t* buffer, size_t max_length,
                                  mip_timeout /*wait_time*/,
                                  bool /*from_cmd*/,
                                  size_t* out_length,
                                  mip_timestamp* timestamp_out) {
    size_t n = 0;
    while (Serial1.available() && n < max_length) {
        int b = Serial1.read();
        if (b < 0) break;
        buffer[n++] = (uint8_t)b;
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
                                   size_t payload_length) {
    if (payload_length > MIP_FIELD_PAYLOAD_LENGTH_MAX) {
        return false;
    }

    uint8_t packet_buffer[MIP_PACKET_LENGTH_MAX];
    mip_packet_view packet;
    mip_packet_create(&packet, packet_buffer, sizeof(packet_buffer), MIP_AIDING_CMD_DESC_SET);
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

// 2. Aiding Measurement Enable (0x13, 0x50)
//    Enables External Position / Velocity / Heading aiding sources
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

// =========================================================================
// MIP REALTIME OUTPUT (3 fields per GNSS epoch)
// =========================================================================
static void try_send_mip() {
    if (!pvt.valid || !pcv.valid || !vcv.valid) { mip_skipped++; return; }
    if (pvt.tow_ms != pcv.tow_ms || pvt.tow_ms != vcv.tow_ms) return;
    if (pvt.mode == 0 || pvt.mode == 3) { mip_skipped++; return; }
    if (pvt.tow_ms == last_mip_tow) return;
    last_mip_tow = pvt.tow_ms;

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
        cmd.valid_flags = MIP_AIDING_POS_LLH_COMMAND_VALID_FLAGS_ALL;

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
        }
    }

    // --- Field 2: External Velocity NED (0x29) ---
    {
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
        cmd.valid_flags = MIP_AIDING_VEL_NED_COMMAND_VALID_FLAGS_ALL;

        mip_serializer s;
        uint8_t buf[80];
        microstrain_serializer_init_insertion(&s, buf, sizeof(buf));
        insert_mip_aiding_vel_ned_command(&s, &cmd);
        if (microstrain_serializer_is_ok(&s)) {
            if (mip_send_command_field(MIP_CMD_DESC_AIDING_VEL_NED,
                                       buf, microstrain_serializer_length(&s))) {
                mip_total++;
            }
        }
    }

    // --- Field 3: External Heading True (0x31) ---
    //     ONLY if dual-antenna heading is valid
    if (att.valid && acv.valid && att.tow_ms == pvt.tow_ms) {
        mip_aiding_heading_true_command cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.time = t;
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
            }
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
        default: return;
    }
    try_send_mip();
}

// =========================================================================
// SBF STREAM STATE MACHINE (unchanged from handwritten version)
// =========================================================================
static uint8_t  sbf_buf[1024];
static uint16_t sbf_idx = 0;
static enum { S_SYNC1, S_SYNC2, S_BODY } sbf_state = S_SYNC1;
static uint16_t sbf_len = 0;

static void feed_gps_byte(uint8_t b) {
    switch (sbf_state) {
        case S_SYNC1:
            if (b == 0x24) { sbf_buf[0] = b; sbf_idx = 1; sbf_state = S_SYNC2; }
            break;
        case S_SYNC2:
            if (b == 0x40) { sbf_buf[1] = b; sbf_idx = 2; sbf_state = S_BODY; sbf_len = 0; }
            else if (b == 0x24) { sbf_buf[0] = b; sbf_idx = 1; }
            else { sbf_state = S_SYNC1; }
            break;
        case S_BODY:
            if (sbf_idx < sizeof(sbf_buf)) sbf_buf[sbf_idx++] = b;
            else { sbf_state = S_SYNC1; sbf_idx = 0; break; }
            if (sbf_idx == 8) {
                memcpy(&sbf_len, &sbf_buf[6], 2);
                if (sbf_len < 8 || sbf_len > sizeof(sbf_buf) || (sbf_len % 4) != 0) {
                    sbf_state = S_SYNC1; sbf_idx = 0;
                }
            } else if (sbf_idx >= 8 && sbf_idx == sbf_len) {
                uint16_t crc_expect;
                memcpy(&crc_expect, &sbf_buf[2], 2);
                uint16_t crc_calc = sbf_crc16(&sbf_buf[4], sbf_len - 4);
                if (crc_calc == crc_expect) {
                    uint16_t blk_id;
                    memcpy(&blk_id, &sbf_buf[4], 2);
                    handle_sbf_packet(blk_id, sbf_buf, sbf_len);
                }
                sbf_state = S_SYNC1; sbf_idx = 0;
            }
            break;
    }
}

// =========================================================================
// STATUS PRINTER (same as before)
// =========================================================================
static uint32_t last_status_ms = 0;
static uint32_t last_pvt_count = 0, last_pcv_count = 0, last_vcv_count = 0;
static uint32_t last_att_count = 0, last_acv_count = 0;
static uint32_t last_mip_count = 0;
static uint32_t last_pps_count = 0;

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
    if (att.valid) {
        Serial.print(F(" Heading         : "));
        Serial.print(att.heading_deg, 2); Serial.print(F(" deg"));
        if (acv.valid) {
            Serial.print(F("  (sigma="));
            Serial.print(acv.std_heading, 2); Serial.print(F(" deg)"));
        }
        Serial.println();
    }
    Serial.println(F("=========================================="));

    last_pvt_count = cnt_pvt;
    last_pcv_count = cnt_poscov;
    last_vcv_count = cnt_velcov;
    last_att_count = cnt_att;
    last_acv_count = cnt_attcov;
    last_mip_count = mip_total;
    last_pps_count = pps_snapshot;
}

// =========================================================================
// SETUP / LOOP
// =========================================================================
void setup() {
    Serial.begin(USB_BAUD);
    Serial2.begin(GNSS_BAUD, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);
    Serial1.begin(IMU_BAUD, SERIAL_8N1, IMU_RX_PIN, IMU_TX_PIN);

    pinMode(PPS_IN_PIN, INPUT);
    pinMode(PPS_OUT_IMU_PIN, OUTPUT);
    digitalWrite(PPS_OUT_IMU_PIN, digitalRead(PPS_IN_PIN));
    attachInterrupt(digitalPinToInterrupt(PPS_IN_PIN), pps_isr, CHANGE);

    uint32_t t0 = millis();
    while (!Serial && (millis() - t0) < 3000) {}

    if (PRINT_STATUS_TO_USB) {
        Serial.println();
        Serial.println(F("================================================"));
        Serial.println(F(" ESP32 SBF -> MIP Bridge (with MIP SDK)"));
        Serial.println(F(" Serial2 RX/TX GPIO25/26: GPS SBF @ 115200"));
        Serial.println(F(" Serial1 TX/RX GPIO32/33: IMU MIP @ 115200"));
        Serial.print(F(" PPS in/out     : Pin "));
        Serial.print(PPS_IN_PIN);
        Serial.print(F(" -> Pin "));
        Serial.println(PPS_OUT_IMU_PIN);
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
    while (Serial2.available()) {
        int b = Serial2.read();
        if (b < 0) break;
        feed_gps_byte((uint8_t)b);
    }

    // Pump MIP SDK (handles any pending replies from CV7)
    mip_interface_update(&mip_dev, 0, false);

    if (PRINT_STATUS_TO_USB && millis() - last_status_ms >= 1000) {
        last_status_ms = millis();
        print_status();
    }
}
