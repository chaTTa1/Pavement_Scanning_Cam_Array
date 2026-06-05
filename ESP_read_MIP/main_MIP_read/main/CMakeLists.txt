idf_component_register(
    SRCS "main_MIP_read.c"
    INCLUDE_DIRS "."
    REQUIRES esp_driver_gpio esp_driver_uart esp_timer mip_sdk
)
