# Install script for directory: C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/mip

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files/verify_mip_project")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "mip" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-build/src/c/mip/Debug/mip.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-build/src/c/mip/Release/mip.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-build/src/c/mip/MinSizeRel/mip.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-build/src/c/mip/RelWithDebInfo/mip.lib")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/microstrain/c/mip" TYPE FILE FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/mip/mip_version.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/microstrain/c/mip" TYPE FILE FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/mip/mip_cmdqueue.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/microstrain/c/mip" TYPE FILE FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/mip/mip_descriptors.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/microstrain/c/mip" TYPE FILE FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/mip/mip_device_models.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/microstrain/c/mip" TYPE FILE FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/mip/mip_dispatch.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/microstrain/c/mip" TYPE FILE FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/mip/mip_field.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/microstrain/c/mip" TYPE FILE FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/mip/mip_interface.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/microstrain/c/mip" TYPE FILE FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/mip/mip_logging.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/microstrain/c/mip" TYPE FILE FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/mip/mip_packet.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/microstrain/c/mip" TYPE FILE FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/mip/mip_parser.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/microstrain/c/mip" TYPE FILE FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/mip/mip_result.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/microstrain/c/mip" TYPE FILE FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/mip/mip_types.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/microstrain/c/mip" TYPE FILE FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/mip/mip_serialization.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/microstrain/c/mip/definitions" TYPE FILE FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/mip/definitions/common.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/microstrain/c/mip" TYPE FILE FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/mip/mip_all.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/microstrain/c/mip/definitions" TYPE FILE FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/mip/definitions/commands_3dm.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/microstrain/c/mip/definitions" TYPE FILE FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/mip/definitions/commands_aiding.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/microstrain/c/mip/definitions" TYPE FILE FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/mip/definitions/commands_base.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/microstrain/c/mip/definitions" TYPE FILE FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/mip/definitions/commands_filter.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/microstrain/c/mip/definitions" TYPE FILE FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/mip/definitions/commands_gnss.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/microstrain/c/mip/definitions" TYPE FILE FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/mip/definitions/commands_rtk.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/microstrain/c/mip/definitions" TYPE FILE FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/mip/definitions/commands_system.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/microstrain/c/mip/definitions" TYPE FILE FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/mip/definitions/data_filter.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/microstrain/c/mip/definitions" TYPE FILE FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/mip/definitions/data_gnss.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/microstrain/c/mip/definitions" TYPE FILE FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/mip/definitions/data_sensor.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/microstrain/c/mip/definitions" TYPE FILE FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/mip/definitions/data_shared.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/microstrain/c/mip/definitions" TYPE FILE FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/mip/definitions/data_system.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "mip" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/mip/cmake" TYPE FILE FILES
    "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-build/src/c/mip/mip-config.cmake"
    "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-build/src/c/mip/mip-config-version.cmake"
    )
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-build/src/c/mip/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
