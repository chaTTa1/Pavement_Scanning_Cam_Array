cmake_minimum_required(VERSION 3.16)

project(cv7_pangolin_viewer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Pangolin REQUIRED)

add_executable(cv7_pangolin_viewer src/main.cpp)
target_include_directories(cv7_pangolin_viewer PRIVATE ${Pangolin_INCLUDE_DIRS})
target_link_libraries(cv7_pangolin_viewer PRIVATE ${Pangolin_LIBRARIES})

if(WIN32)
    target_link_libraries(cv7_pangolin_viewer PRIVATE ws2_32)
endif()

if(MSVC)
    target_compile_options(cv7_pangolin_viewer PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(cv7_pangolin_viewer PRIVATE -Wall -Wextra -Wpedantic)
endif()

