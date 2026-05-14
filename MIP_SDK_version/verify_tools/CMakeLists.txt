cmake_minimum_required(VERSION 3.20)
project(verify_mip_project LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

include(FetchContent)

set(MICROSTRAIN_ENABLE_CPP OFF CACHE BOOL "" FORCE)
set(MICROSTRAIN_ENABLE_SERIAL OFF CACHE BOOL "" FORCE)
set(MICROSTRAIN_ENABLE_TCP OFF CACHE BOOL "" FORCE)
set(MICROSTRAIN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MICROSTRAIN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MIP_ENABLE_METADATA OFF CACHE BOOL "" FORCE)
set(MIP_ENABLE_EXTRAS OFF CACHE BOOL "" FORCE)
set(MIP_ENABLE_DIAGNOSTICS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    mip_sdk
    GIT_REPOSITORY https://github.com/HBK-MicroStrain/mip_sdk.git
    GIT_TAG main
)

FetchContent_MakeAvailable(mip_sdk)

add_executable(verify_mip verify_mip_updated.cpp)

target_link_libraries(verify_mip PRIVATE
    mip
    microstrain
)

target_include_directories(verify_mip PRIVATE
    ${mip_sdk_SOURCE_DIR}/src/c
    ${mip_sdk_SOURCE_DIR}/src/cpp
)
