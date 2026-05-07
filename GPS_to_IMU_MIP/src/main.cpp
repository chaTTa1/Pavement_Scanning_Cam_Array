/*
 * Teensy 4.1  SBF -> MIP Bridge   (PlatformIO / pure C++)
 * ========================================================
 * Hardware:
 *   simpleRTK3B Heading (Mosaic-X5)  --Arduino rail TX1-->  Teensy Pin 0 (Serial1 RX)
 *                                     IOREF switch = 3.3V (mandatory!)
 *                                     XBee socket: WiFi NTRIP Master (COM2)
 *                                     USB-C: standalone power
 *
 *   Teensy Pin 8 (Serial2 TX)  -->  CV7-INS Pin 4 (RxD, main UART)
 *   Teensy Pin 7 (Serial2 RX)  <--  CV7-INS Pin 5 (TxD)
 *   Teensy 3.3V                -->  CV7-INS Pin 3 (Vin)
 *   Teensy GND                 ---  CV7-INS Pin 8 (GND)
 *
 * SBF inputs (configure on simpleRTK3B side):
 *   Stream1 -> COM1 @ 115200, msec100, blocks:
 *     PVTGeodetic   (4007)    position + velocity + mode + sats
 *     PosCovGeodetic(5906)    position covariance
 *     VelCovGeodetic(5908)    velocity covariance
 *     AttEuler      (5938)    heading + pitch + roll  (Heading board only)
 *     AttCovEuler   (5939)    attitude covariance
 *
 * MIP outputs to CV7-INS (External Aiding, descriptor set 0x13):
 *   0x62 External GNSS Time   (12 bytes payload)
 *   0x16 External Position LLH(48 bytes payload)
 *   0x17 External Velocity NED(36 bytes payload)
 *   0x28 External Heading True(20 bytes payload, only when Heading is valid)
 *
 * Notes:
 *   - All MIP fields use BIG-endian byte order (manual byte swapping required)
 *   - SBF fields are LITTLE-endian (matches Teensy native order, direct memcpy ok)
 *   - "Do-Not-Use" sentinel values in SBF (-2e10 / -2e9) are detected and skipped
 *   - One MIP packet per GPS epoch (deduplicated by TOW)
 */

#include <Arduino.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// =========================================================================
// CONFIG
// =========================================================================
static constexpr uint32_t USB_BAUD  = 115200;
static constexpr uint32_t GNSS_BAUD = 115200;
static constexpr uint32_t IMU_BAUD  = 115200;

// Larger Serial1 RX buffer to absorb SBF bursts at 10+ Hz
static uint8_t serial1_rx_buf[4096];

// =========================================================================
// MIP PROTOCOL  (manual implementation, big-endian, Fletcher-16)
// =========================================================================
static constexpr uint8_t MIP_SYNC1 = 0x75;
static constexpr uint8_t MIP_SYNC2 = 0x65;

static constexpr uint8_t DESC_SET_AIDING       = 0x13;
static constexpr uint8_t FIELD_EXT_POS_LLH     = 0x16;
static constexpr uint8_t FIELD_EXT_VEL_NED     = 0x17;
static constexpr uint8_t FIELD_EXT_HEADING_TRUE= 0x28;
static constexpr uint8_t FIELD_EXT_GNSS_TIME   = 0x62;

// Fletcher-16 over (sync-stripped) header + payload
static void fletcher16(const uint8_t* data, size_t len, uint8_t& c0, uint8_t& c1) {
    c0 = 0; c1 = 0;
    for (size_t i = 0; i < len; ++i) {
        c0 = (uint8_t)(c0 + data[i]);
        c1 = (uint8_t)(c1 + c0);
    }
}

// Big-endian writers
static inline void put_be16(uint8_t* p, uint16_t v) {
    p[0] = (uint8_t)(v >> 8); p[1] = (uint8_t)(v & 0xFF);
}
static inline void put_be32(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);  p[3] = (uint8_t)(v & 0xFF);
}
static inline void put_be32f(uint8_t* p, float v) {
    uint32_t u; memcpy(&u, &v, 4); put_be32(p, u);
}
static inline void put_be64(uint8_t* p, uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = (uint8_t)(v >> (56 - 8*i));
}
static inline void put_be64d(uint8_t* p, double v) {
    uint64_t u; memcpy(&u, &v, 8); put_be64(p, u);
}

// 0x13, 0x62  External GNSS Time (12 bytes)
static size_t pack_ext_gnss_time(uint8_t* buf, double tow_s, uint16_t week,
                                 uint16_t valid_flags = 0x0003) {
    put_be64d(buf + 0,  tow_s);
    put_be16 (buf + 8,  week);
    put_be16 (buf + 10, valid_flags);
    return 12;
}

// 0x13, 0x16  External Position LLH (48 bytes)
static size_t pack_ext_pos_llh(uint8_t* buf,
                               double tow_s, uint16_t week,
                               double lat_deg, double lon_deg, double height_m,
                               float unc_n, float unc_e, float unc_d,
                               uint16_t valid_flags = 0x000F) {
    put_be64d(buf + 0,  tow_s);
    put_be16 (buf + 8,  week);
    put_be64d(buf + 10, lat_deg);
    put_be64d(buf + 18, lon_deg);
    put_be64d(buf + 26, height_m);
    put_be32f(buf + 34, unc_n);
    put_be32f(buf + 38, unc_e);
    put_be32f(buf + 42, unc_d);
    put_be16 (buf + 46, valid_flags);
    return 48;
}

// 0x13, 0x17  External Velocity NED (36 bytes)
static size_t pack_ext_vel_ned(uint8_t* buf,
                               double tow_s, uint16_t week,
                               float vn, float ve, float vd,
                               float unc_n, float unc_e, float unc_d,
                               uint16_t valid_flags = 0x0007) {
    put_be64d(buf + 0,  tow_s);
    put_be16 (buf + 8,  week);
    put_be32f(buf + 10, vn);
    put_be32f(buf + 14, ve);
    put_be32f(buf + 18, vd);
    put_be32f(buf + 22, unc_n);
    put_be32f(buf + 26, unc_e);
    put_be32f(buf + 30, unc_d);
    put_be16 (buf + 34, valid_flags);
    return 36;
}

// 0x13, 0x28  External Heading True (20 bytes)
static size_t pack_ext_heading_true(uint8_t* buf,
                                    double tow_s, uint16_t week,
                                    float heading_deg, float heading_unc_deg,
                                    uint16_t valid_flags = 0x0003) {
    put_be64d(buf + 0,  tow_s);
    put_be16 (buf + 8,  week);
    put_be32f(buf + 10, heading_deg);
    put_be32f(buf + 14, heading_unc_deg);
    put_be16 (buf + 18, valid_flags);
    return 20;
}

// Build & send a MIP packet with multiple fields on Serial2
struct MipField {
    uint8_t  desc;
    uint8_t  data[64];
    uint8_t  len;     // payload length (NOT including the 2-byte field header)
};

static bool send_mip_packet(uint8_t desc_set, const MipField* fields, uint8_t n_fields) {
    uint8_t pkt[256];
    uint16_t idx = 0;

    pkt[idx++] = MIP_SYNC1;
    pkt[idx++] = MIP_SYNC2;
    pkt[idx++] = desc_set;
    uint16_t len_pos = idx;
    pkt[idx++] = 0;  // payload length, fill later

    for (uint8_t i = 0; i < n_fields; ++i) {
        const MipField& f = fields[i];
        uint8_t flen = f.len + 2;
        pkt[idx++] = flen;
        pkt[idx++] = f.desc;
        memcpy(&pkt[idx], f.data, f.len);
        idx += f.len;
    }
    pkt[len_pos] = (uint8_t)(idx - len_pos - 1);  // payload length

    uint8_t c0, c1;
    fletcher16(&pkt[2], idx - 2, c0, c1);
    pkt[idx++] = c0;
    pkt[idx++] = c1;

    Serial2.write(pkt, idx);
    return true;
}

// =========================================================================
// SBF PARSER
// =========================================================================
static constexpr uint16_t SBF_PVT_GEODETIC      = 4007;
static constexpr uint16_t SBF_POS_COV_GEODETIC  = 5906;
static constexpr uint16_t SBF_VEL_COV_GEODETIC  = 5908;
static constexpr uint16_t SBF_ATT_EULER         = 5938;
static constexpr uint16_t SBF_ATT_COV_EULER     = 5939;

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

// SBF "Do-Not-Use" sentinel detection
static inline bool is_dnu_d(double v) { return v <= -1.999e10; }   // -2e10
static inline bool is_dnu_f(float  v) { return v <= -1.999e9f; }   // -2e9

// Parsed snapshots (latest values)
struct PvtSnapshot {
    bool     valid       = false;
    uint32_t tow_ms      = 0;
    uint16_t week        = 0;
    uint8_t  mode        = 0;        // 0=No PVT, 1=Stand-Alone, 4=RTK Fixed, 5=RTK Float
    uint8_t  err         = 0;
    double   lat_deg     = 0;
    double   lon_deg     = 0;
    double   h_m         = 0;
    float    vn          = 0;
    float    ve          = 0;
    float    vu          = 0;        // Up (SBF)
    uint8_t  num_sats    = 0;
};
struct PosCovSnap {
    bool   valid = false;
    uint32_t tow_ms = 0;
    float  std_n = 0, std_e = 0, std_u = 0;
};
struct VelCovSnap {
    bool   valid = false;
    uint32_t tow_ms = 0;
    float  std_vn = 0, std_ve = 0, std_vu = 0;
};
struct AttSnap {
    bool   valid = false;
    uint32_t tow_ms = 0;
    float  heading_deg = 0;
    float  pitch_deg = 0;
    float  roll_deg = 0;
};
struct AttCovSnap {
    bool   valid = false;
    float  std_heading = 0;
    float  std_pitch = 0;
    float  std_roll = 0;
};

static PvtSnapshot pvt;
static PosCovSnap  pcv;
static VelCovSnap  vcv;
static AttSnap     att;
static AttCovSnap  acv;

static uint32_t cnt_pvt = 0, cnt_poscov = 0, cnt_velcov = 0, cnt_att = 0, cnt_attcov = 0;
static uint32_t mip_total = 0, mip_skipped = 0;
static uint32_t last_mip_tow = 0xFFFFFFFFu;

// SBF block parsers (block payload starts at offset 8 within the raw SBF block)
// All multi-byte values in SBF are little-endian (matches Teensy native).

static void parse_pvt_geodetic(const uint8_t* blk, uint16_t /*len*/) {
    // PVTGeodetic v2
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

static void parse_pos_cov_geodetic(const uint8_t* blk, uint16_t /*len*/) {
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

static void parse_vel_cov_geodetic(const uint8_t* blk, uint16_t /*len*/) {
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

static void parse_att_euler(const uint8_t* blk, uint16_t /*len*/) {
    AttSnap s;
    memcpy(&s.tow_ms, blk + 8, 4);
    memcpy(&s.heading_deg, blk + 16, 4);
    memcpy(&s.pitch_deg,   blk + 20, 4);
    memcpy(&s.roll_deg,    blk + 24, 4);
    s.valid = !is_dnu_f(s.heading_deg);
    att = s;
    cnt_att++;
}

static void parse_att_cov_euler(const uint8_t* blk, uint16_t /*len*/) {
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
// MIP BUILD AND SEND  (called when an epoch's PVT+PosCov+VelCov are ready)
// =========================================================================
static void try_send_mip() {
    // Need at least PVT+PosCov+VelCov, all from same epoch
    if (!pvt.valid || !pcv.valid || !vcv.valid) { mip_skipped++; return; }
    if (pvt.tow_ms != pcv.tow_ms || pvt.tow_ms != vcv.tow_ms) {
        return;  // wait for them to align
    }
    // Only send for "useful" PVT modes (skip No PVT)
    if (pvt.mode == 0 || pvt.mode == 3) { mip_skipped++; return; }
    // Dedup: don't re-send the same epoch
    if (pvt.tow_ms == last_mip_tow) return;
    last_mip_tow = pvt.tow_ms;

    double tow_s = pvt.tow_ms * 1e-3;

    MipField fields[4];
    uint8_t  nf = 0;

    // Time
    fields[nf].desc = FIELD_EXT_GNSS_TIME;
    fields[nf].len  = (uint8_t)pack_ext_gnss_time(fields[nf].data, tow_s, pvt.week);
    nf++;

    // Position
    fields[nf].desc = FIELD_EXT_POS_LLH;
    fields[nf].len  = (uint8_t)pack_ext_pos_llh(fields[nf].data,
                                                tow_s, pvt.week,
                                                pvt.lat_deg, pvt.lon_deg, pvt.h_m,
                                                pcv.std_n, pcv.std_e, pcv.std_u);
    nf++;

    // Velocity (NED: D = -U)
    fields[nf].desc = FIELD_EXT_VEL_NED;
    fields[nf].len  = (uint8_t)pack_ext_vel_ned(fields[nf].data,
                                                tow_s, pvt.week,
                                                pvt.vn, pvt.ve, -pvt.vu,
                                                vcv.std_vn, vcv.std_ve, vcv.std_vu);
    nf++;

    // Heading (only if valid + same epoch)
    if (att.valid && acv.valid && att.tow_ms == pvt.tow_ms) {
        fields[nf].desc = FIELD_EXT_HEADING_TRUE;
        fields[nf].len  = (uint8_t)pack_ext_heading_true(fields[nf].data,
                                                         tow_s, pvt.week,
                                                         att.heading_deg,
                                                         acv.std_heading);
        nf++;
    }

    send_mip_packet(DESC_SET_AIDING, fields, nf);
    mip_total++;
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
// SBF STREAM STATE MACHINE
// =========================================================================
static uint8_t  sbf_buf[1024];
static uint16_t sbf_idx = 0;
static enum { S_SYNC1, S_SYNC2, S_BODY } sbf_state = S_SYNC1;
static uint16_t sbf_len = 0;

static void feed_serial1_byte(uint8_t b) {
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
            // After 8 bytes we have header (sync2 + crc2 + id2 + length2)
            if (sbf_idx == 8) {
                memcpy(&sbf_len, &sbf_buf[6], 2);
                if (sbf_len < 8 || sbf_len > sizeof(sbf_buf) || (sbf_len % 4) != 0) {
                    sbf_state = S_SYNC1; sbf_idx = 0;
                }
            } else if (sbf_idx >= 8 && sbf_idx == sbf_len) {
                // Got complete block. Verify CRC.
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
// STATUS PRINTER
// =========================================================================
static uint32_t last_status_ms = 0;
static uint32_t last_pvt_count = 0, last_pcv_count = 0, last_vcv_count = 0;
static uint32_t last_att_count = 0, last_acv_count = 0;
static uint32_t last_mip_count = 0;

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
    Serial.println();
    Serial.println(F("================ [STATUS] ================"));
    Serial.print  (F(" uptime          : ")); Serial.print(millis()/1000); Serial.println(F(" s"));
    Serial.print  (F(" SBF rates       : "));
    Serial.print(F("PVT="));    Serial.print(cnt_pvt    - last_pvt_count);
    Serial.print(F(" PosCov="));Serial.print(cnt_poscov - last_pcv_count);
    Serial.print(F(" VelCov="));Serial.print(cnt_velcov - last_vcv_count);
    Serial.print(F(" Att="));   Serial.print(cnt_att    - last_att_count);
    Serial.print(F(" AttCov="));Serial.print(cnt_attcov - last_acv_count);
    Serial.println(F(" /s"));
    Serial.print  (F(" MIP rate        : "));
    Serial.print(mip_total - last_mip_count); Serial.print(F(" /s   total: "));
    Serial.print(mip_total); Serial.print(F("   skipped: "));
    Serial.println(mip_skipped);
    Serial.println(F(" ---- LATEST PVT ----"));
    if (pvt.valid) {
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
        Serial.print(F(" Vel NED         : "));
        Serial.print(pvt.vn, 3); Serial.print(F("  "));
        Serial.print(pvt.ve, 3); Serial.print(F("  "));
        Serial.print(-pvt.vu, 3);Serial.println(F(" m/s"));
        if (pcv.valid) {
            Serial.print(F(" sigma N/E/U     : "));
            Serial.print(pcv.std_n, 3); Serial.print(F(" "));
            Serial.print(pcv.std_e, 3); Serial.print(F(" "));
            Serial.print(pcv.std_u, 3); Serial.println(F(" m"));
        }
        Serial.print(F(" Sat used        : ")); Serial.println(pvt.num_sats);
    } else {
        Serial.println(F(" Mode            : 0 (No PVT)"));
        Serial.println(F(" (waiting for valid GNSS fix - connect antenna)"));
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
    if (pvt.mode == 4 && att.valid && mip_total > 0) {
        Serial.println(F(" >>> RTK Fixed + Heading + MIP streaming to CV7-INS <<<"));
    }
    Serial.println(F("=========================================="));

    last_pvt_count = cnt_pvt;
    last_pcv_count = cnt_poscov;
    last_vcv_count = cnt_velcov;
    last_att_count = cnt_att;
    last_acv_count = cnt_attcov;
    last_mip_count = mip_total;
}

// =========================================================================
// SETUP / LOOP
// =========================================================================
void setup() {
    Serial.begin(USB_BAUD);
    Serial1.begin(GNSS_BAUD);
    Serial2.begin(IMU_BAUD);
    Serial1.addMemoryForRead(serial1_rx_buf, sizeof(serial1_rx_buf));

    uint32_t t0 = millis();
    while (!Serial && (millis() - t0) < 3000) {}

    Serial.println();
    Serial.println(F("================================================"));
    Serial.println(F(" Teensy 4.1  SBF -> MIP Bridge"));
    Serial.println(F(" Serial1 (Pin 0): simpleRTK3B SBF @ 115200"));
    Serial.println(F(" Serial2 (Pin 8): CV7-INS MIP    @ 115200"));
    Serial.println(F("================================================"));

    last_status_ms = millis();
}

void loop() {
    // Drain Serial1 -> SBF parser -> auto MIP build/send
    while (Serial1.available()) {
        int b = Serial1.read();
        if (b < 0) break;
        feed_serial1_byte((uint8_t)b);
    }

    // 1 Hz status to USB
    if (millis() - last_status_ms >= 1000) {
        last_status_ms = millis();
        print_status();
    }
}