# Install script for directory: C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/microstrain

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

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-build/src/c/microstrain/connections/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "microstrain" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-build/src/c/microstrain/Debug/microstrain.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-build/src/c/microstrain/Release/microstrain.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-build/src/c/microstrain/MinSizeRel/microstrain.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-build/src/c/microstrain/RelWithDebInfo/microstrain.lib")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/microstrain/c/microstrain" TYPE FILE FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/microstrain/embedded_time.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/microstrain/c/microstrain" TYPE FILE FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/microstrain/logging.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/microstrain/c/microstrain" TYPE FILE FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/microstrain/platform.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/microstrain/c/microstrain" TYPE FILE FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/microstrain/serialization.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/microstrain/c/microstrain" TYPE FILE FILES "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src/src/c/microstrain/strings.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "microstrain" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/microstrain/cmake" TYPE FILE FILES
    "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-build/src/c/microstrain/microstrain-config.cmake"
    "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-build/src/c/microstrain/microstrain-config-version.cmake"
    )
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-build/src/c/microstrain/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
