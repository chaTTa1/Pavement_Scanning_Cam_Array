set(MIP_SDK_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../MIP_SDK_version/MIP_sdk_version/lib/mip_sdk/src")

file(GLOB MIP_CORE_SOURCES
    "${MIP_SDK_ROOT}/mip/*.c"
)

file(GLOB MIP_DEFINITION_SOURCES
    "${MIP_SDK_ROOT}/mip/definitions/*.c"
)

set(MICROSTRAIN_SOURCES
    "${MIP_SDK_ROOT}/microstrain/logging.c"
    "${MIP_SDK_ROOT}/microstrain/serialization.c"
    "${MIP_SDK_ROOT}/microstrain/strings.c"
)

idf_component_register(
    SRCS
        ${MIP_CORE_SOURCES}
        ${MIP_DEFINITION_SOURCES}
        ${MICROSTRAIN_SOURCES}
    INCLUDE_DIRS
        "${MIP_SDK_ROOT}"
)

target_compile_definitions(${COMPONENT_LIB} PUBLIC
    MICROSTRAIN_PLATFORM_EMBEDDED=1
    MIP_USE_EXTRAS=0
)

target_compile_options(${COMPONENT_LIB} PRIVATE
    -Wno-missing-field-initializers
)
