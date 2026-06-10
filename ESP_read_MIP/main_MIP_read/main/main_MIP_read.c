#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "driver/gpio.h"
#include "driver/uart.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "microstrain/serialization.h"
#include "mip/mip_all.h"
#include "mip/definitions/commands_aiding.h"

static const char* TAG = "main_MIP_read";

// ESP32-WROVER modules often reserve GPIO16/GPIO17 for PSRAM.
// Connect Teensy Pin 8 (Serial2 TX) to this ESP32 RX pin, and connect GND to GND.
#define MIP_UART_NUM       UART_NUM_2
#define MIP_UART_RX_GPIO   GPIO_NUM_34
#define MIP_UART_BAUD      115200
#define MIP_UART_BUF_SIZE  2048

#define PPS_RX_GPIO        GPIO_NUM_35
#define PPS_LED_GPIO       GPIO_NUM_2
#define PPS_LED_FLASH_US   200000LL

#define DEG_PER_RAD        57.2957795130823208768
#define STATS_PERIOD_US    1000000LL

typedef struct {
    uint32_t rx_bytes;
    uint32_t packets;
    uint32_t invalid_packets;
    uint32_t wrong_desc_set;
    uint32_t pos_count;
    uint32_t vel_count;
    uint32_t heading_count;
    uint32_t decode_errors;
    uint32_t other_fields;
} mip_stats_t;

static mip_parser g_parser;
static mip_stats_t g_stats;
static int64_t g_last_stats_us = 0;
static volatile bool g_pps_led_active = false;
static volatile int64_t g_pps_led_off_us = 0;

static const char* yes_no(bool value)
{
    return value ? "yes" : "no";
}

static uint32_t now_ms(void)
{
    return (uint32_t)(esp_timer_get_time() / 1000ULL);
}

static bool extract_pos_llh(const mip_field_view* field, mip_aiding_pos_llh_command* out)
{
    memset(out, 0, sizeof(*out));
    microstrain_serializer serializer;
    microstrain_serializer_init_extraction(&serializer,
                                           mip_field_payload(field),
                                           mip_field_payload_length(field));
    extract_mip_aiding_pos_llh_command(&serializer, out);
    return microstrain_serializer_is_ok(&serializer)
        && microstrain_serializer_is_complete(&serializer);
}

static bool extract_vel_ned(const mip_field_view* field, mip_aiding_vel_ned_command* out)
{
    memset(out, 0, sizeof(*out));
    microstrain_serializer serializer;
    microstrain_serializer_init_extraction(&serializer,
                                           mip_field_payload(field),
                                           mip_field_payload_length(field));
    extract_mip_aiding_vel_ned_command(&serializer, out);
    return microstrain_serializer_is_ok(&serializer)
        && microstrain_serializer_is_complete(&serializer);
}

static bool extract_heading_true(const mip_field_view* field, mip_aiding_heading_true_command* out)
{
    memset(out, 0, sizeof(*out));
    microstrain_serializer serializer;
    microstrain_serializer_init_extraction(&serializer,
                                           mip_field_payload(field),
                                           mip_field_payload_length(field));
    extract_mip_aiding_heading_true_command(&serializer, out);
    return microstrain_serializer_is_ok(&serializer)
        && microstrain_serializer_is_complete(&serializer);
}

static void print_pos_llh(const mip_aiding_pos_llh_command* pos)
{
    const bool lat_valid = (pos->valid_flags & MIP_AIDING_POS_LLH_COMMAND_VALID_FLAGS_LATITUDE) != 0;
    const bool lon_valid = (pos->valid_flags & MIP_AIDING_POS_LLH_COMMAND_VALID_FLAGS_LONGITUDE) != 0;
    const bool h_valid = (pos->valid_flags & MIP_AIDING_POS_LLH_COMMAND_VALID_FLAGS_HEIGHT) != 0;

    ESP_LOGI(TAG,
             "POS frame=%u flags=0x%04X valid(lat/lon/h)=%s/%s/%s "
             "lat=%.8f lon=%.8f h=%.3f sigmaN/E/U=%.3f/%.3f/%.3f timebase=%u ns=%llu",
             (unsigned)pos->frame_id,
             (unsigned)pos->valid_flags,
             yes_no(lat_valid),
             yes_no(lon_valid),
             yes_no(h_valid),
             pos->latitude,
             pos->longitude,
             pos->height,
             (double)pos->uncertainty[0],
             (double)pos->uncertainty[1],
             (double)pos->uncertainty[2],
             (unsigned)pos->time.timebase,
             (unsigned long long)pos->time.nanoseconds);
}

static void print_vel_ned(const mip_aiding_vel_ned_command* vel)
{
    const bool vn_valid = (vel->valid_flags & MIP_AIDING_VEL_NED_COMMAND_VALID_FLAGS_X) != 0;
    const bool ve_valid = (vel->valid_flags & MIP_AIDING_VEL_NED_COMMAND_VALID_FLAGS_Y) != 0;
    const bool vd_valid = (vel->valid_flags & MIP_AIDING_VEL_NED_COMMAND_VALID_FLAGS_Z) != 0;

    ESP_LOGI(TAG,
             "VEL frame=%u flags=0x%04X valid(N/E/D)=%s/%s/%s "
             "vn=%.3f ve=%.3f vd=%.3f sigmaN/E/D=%.3f/%.3f/%.3f timebase=%u ns=%llu",
             (unsigned)vel->frame_id,
             (unsigned)vel->valid_flags,
             yes_no(vn_valid),
             yes_no(ve_valid),
             yes_no(vd_valid),
             (double)vel->velocity[0],
             (double)vel->velocity[1],
             (double)vel->velocity[2],
             (double)vel->uncertainty[0],
             (double)vel->uncertainty[1],
             (double)vel->uncertainty[2],
             (unsigned)vel->time.timebase,
             (unsigned long long)vel->time.nanoseconds);
}

static void print_heading_true(const mip_aiding_heading_true_command* heading)
{
    ESP_LOGI(TAG,
             "HDG frame=%u flags=0x%04X valid=%s heading=%.2f deg sigma=%.2f deg timebase=%u ns=%llu",
             (unsigned)heading->frame_id,
             (unsigned)heading->valid_flags,
             yes_no((heading->valid_flags & 0x0001U) != 0),
             (double)heading->heading * DEG_PER_RAD,
             (double)heading->uncertainty * DEG_PER_RAD,
             (unsigned)heading->time.timebase,
             (unsigned long long)heading->time.nanoseconds);
}

static void process_aiding_field(const mip_field_view* field)
{
    switch (mip_field_field_descriptor(field)) {
        case MIP_CMD_DESC_AIDING_POS_LLH: {
            mip_aiding_pos_llh_command pos;
            if (extract_pos_llh(field, &pos)) {
                g_stats.pos_count++;
                print_pos_llh(&pos);
            } else {
                g_stats.decode_errors++;
                ESP_LOGW(TAG, "Failed to decode POS LLH field");
            }
            break;
        }
        case MIP_CMD_DESC_AIDING_VEL_NED: {
            mip_aiding_vel_ned_command vel;
            if (extract_vel_ned(field, &vel)) {
                g_stats.vel_count++;
                print_vel_ned(&vel);
            } else {
                g_stats.decode_errors++;
                ESP_LOGW(TAG, "Failed to decode VEL NED field");
            }
            break;
        }
        case MIP_CMD_DESC_AIDING_HEADING_TRUE: {
            mip_aiding_heading_true_command heading;
            if (extract_heading_true(field, &heading)) {
                g_stats.heading_count++;
                print_heading_true(&heading);
            } else {
                g_stats.decode_errors++;
                ESP_LOGW(TAG, "Failed to decode HEADING TRUE field");
            }
            break;
        }
        default:
            g_stats.other_fields++;
            ESP_LOGD(TAG, "Ignoring aiding field descriptor 0x%02X",
                     (unsigned)mip_field_field_descriptor(field));
            break;
    }
}

static bool handle_mip_packet(void* user, const mip_packet_view* packet, mip_timestamp timestamp)
{
    (void)user;
    (void)timestamp;

    g_stats.packets++;
    if (!mip_packet_is_valid(packet)) {
        g_stats.invalid_packets++;
        return true;
    }

    if (mip_packet_descriptor_set(packet) != MIP_AIDING_CMD_DESC_SET) {
        g_stats.wrong_desc_set++;
        ESP_LOGD(TAG, "Ignoring descriptor set 0x%02X",
                 (unsigned)mip_packet_descriptor_set(packet));
        return true;
    }

    mip_field_view field = mip_field_first_from_packet(packet);
    while (mip_field_is_valid(&field)) {
        process_aiding_field(&field);
        mip_field_next(&field);
    }
    return true;
}

static void print_stats_if_due(void)
{
    const int64_t now_us = esp_timer_get_time();
    if ((now_us - g_last_stats_us) < STATS_PERIOD_US) {
        return;
    }
    g_last_stats_us = now_us;

    ESP_LOGI(TAG,
             "STATS rx_bytes=%lu packets=%lu invalid=%lu wrong_set=%lu "
             "pos=%lu vel=%lu heading=%lu decode_errors=%lu other_fields=%lu",
             (unsigned long)g_stats.rx_bytes,
             (unsigned long)g_stats.packets,
             (unsigned long)g_stats.invalid_packets,
             (unsigned long)g_stats.wrong_desc_set,
             (unsigned long)g_stats.pos_count,
             (unsigned long)g_stats.vel_count,
             (unsigned long)g_stats.heading_count,
             (unsigned long)g_stats.decode_errors,
             (unsigned long)g_stats.other_fields);
}

static void IRAM_ATTR pps_isr_handler(void* arg)
{
    (void)arg;
    const int64_t now_us = esp_timer_get_time();
    g_pps_led_off_us = now_us + PPS_LED_FLASH_US;
    g_pps_led_active = true;
    gpio_set_level(PPS_LED_GPIO, 1);
}

static void service_pps_led(void)
{
    if (!g_pps_led_active) {
        return;
    }

    const int64_t now_us = esp_timer_get_time();
    if (now_us >= g_pps_led_off_us) {
        gpio_set_level(PPS_LED_GPIO, 0);
        g_pps_led_active = false;
    }
}

static void configure_pps_led(void)
{
    const gpio_config_t led_config = {
        .pin_bit_mask = 1ULL << PPS_LED_GPIO,
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&led_config));
    gpio_set_level(PPS_LED_GPIO, 0);

    const gpio_config_t pps_config = {
        .pin_bit_mask = 1ULL << PPS_RX_GPIO,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_POSEDGE,
    };
    ESP_ERROR_CHECK(gpio_config(&pps_config));
    ESP_ERROR_CHECK(gpio_install_isr_service(0));
    ESP_ERROR_CHECK(gpio_isr_handler_add(PPS_RX_GPIO, pps_isr_handler, NULL));
}

static void configure_uart(void)
{
    const uart_config_t uart_config = {
        .baud_rate = MIP_UART_BAUD,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };

    ESP_ERROR_CHECK(uart_driver_install(MIP_UART_NUM, MIP_UART_BUF_SIZE, 0, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(MIP_UART_NUM, &uart_config));
    ESP_ERROR_CHECK(uart_set_pin(MIP_UART_NUM,
                                 UART_PIN_NO_CHANGE,
                                 MIP_UART_RX_GPIO,
                                 UART_PIN_NO_CHANGE,
                                 UART_PIN_NO_CHANGE));
}

void app_main(void)
{
    ESP_LOGI(TAG, "ESP32-WROVER-E MIP reader starting");
    ESP_LOGI(TAG, "Wire Teensy Pin 8 TX -> ESP32 GPIO%d RX, and GND -> GND",
             (int)MIP_UART_RX_GPIO);
    ESP_LOGI(TAG, "Wire Teensy Pin 3 PPS -> ESP32 GPIO%d; PPS LED GPIO%d flashes",
             (int)PPS_RX_GPIO, (int)PPS_LED_GPIO);
    ESP_LOGI(TAG, "UART%d baud=%d", (int)MIP_UART_NUM, MIP_UART_BAUD);

    configure_pps_led();
    configure_uart();
    mip_parser_init(&g_parser, handle_mip_packet, NULL, mip_timeout_from_baudrate(MIP_UART_BAUD));

    uint8_t buffer[256];
    g_last_stats_us = esp_timer_get_time();

    while (true) {
        const int n = uart_read_bytes(MIP_UART_NUM, buffer, sizeof(buffer), pdMS_TO_TICKS(100));
        if (n > 0) {
            g_stats.rx_bytes += (uint32_t)n;
            mip_parser_parse(&g_parser, buffer, (size_t)n, now_ms());
        }
        service_pps_led();
        print_stats_if_due();
    }
}
