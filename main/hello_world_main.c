// ================= HARDWIRED PPS GENERATOR =================
// Purpose: Immediately starts pulsing GPIO 18 at 1Hz upon power-up.
// No Wi-Fi, No UDP, No Commands. Just pure hardware signals.
// ===========================================================

#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_timer.h"
#include "esp_log.h"

// ---- CONFIGURATION ----
#define PPS_OUT_GPIO       18           // Output Pin
#define PPS_PERIOD_US      1000000      // 1 second (1Hz)
#define PPS_PULSE_WIDTH_US 10000       // 100 ms width (Easy to see on Scope)
// -----------------------

static const char *TAG = "PPS_HARDWIRE";

// Timers
static esp_timer_handle_t g_pps_periodic_timer = NULL;
static esp_timer_handle_t g_pps_pulse_off_timer = NULL;

// --- 1. Turn Pin LOW (End of Pulse) ---
static void pps_pulse_off_cb(void *arg)
{
    gpio_set_level(PPS_OUT_GPIO, 0);
}

// --- 2. Turn Pin HIGH (Start of Pulse) ---
static void pps_periodic_cb(void *arg)
{
    gpio_set_level(PPS_OUT_GPIO, 1);
    
    // Schedule it to turn off after 100ms
    esp_timer_start_once(g_pps_pulse_off_timer, PPS_PULSE_WIDTH_US);
}

// --- 3. Setup GPIO ---
static void pps_gpio_init(void)
{
    gpio_config_t io_conf = {
        .pin_bit_mask = 1ULL << PPS_OUT_GPIO,
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE
    };
    gpio_config(&io_conf);
    gpio_set_level(PPS_OUT_GPIO, 0);
}

// --- MAIN LOOP ---
void app_main(void)
{
    ESP_LOGI(TAG, "PPS Generator Starting...");

    // 1. Setup the Pin
    pps_gpio_init();

    // 2. Create the "Turn Off" Timer (One-shot)
    esp_timer_create_args_t off_args = {
        .callback = &pps_pulse_off_cb,
        .name = "pps_off"
    };
    ESP_ERROR_CHECK(esp_timer_create(&off_args, &g_pps_pulse_off_timer));

    // 3. Create the "Turn On" Timer (Periodic 1Hz)
    esp_timer_create_args_t periodic_args = {
        .callback = &pps_periodic_cb,
        .name = "pps_periodic"
    };
    ESP_ERROR_CHECK(esp_timer_create(&periodic_args, &g_pps_periodic_timer));

    // 4. START NOW
    ESP_ERROR_CHECK(esp_timer_start_periodic(g_pps_periodic_timer, PPS_PERIOD_US));
    
    ESP_LOGI(TAG, "PPS Running on GPIO %d. Period: %dus, Width: %dus", 
             PPS_OUT_GPIO, PPS_PERIOD_US, PPS_PULSE_WIDTH_US);

    // 5. Keep main alive (just in case)
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(10000)); 
    }
}