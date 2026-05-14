# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src")
  file(MAKE_DIRECTORY "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-src")
endif()
file(MAKE_DIRECTORY
  "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-build"
  "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-subbuild/mip_sdk-populate-prefix"
  "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-subbuild/mip_sdk-populate-prefix/tmp"
  "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-subbuild/mip_sdk-populate-prefix/src/mip_sdk-populate-stamp"
  "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-subbuild/mip_sdk-populate-prefix/src"
  "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-subbuild/mip_sdk-populate-prefix/src/mip_sdk-populate-stamp"
)

set(configSubDirs Debug)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-subbuild/mip_sdk-populate-prefix/src/mip_sdk-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "C:/Users/rqy19/Documents/github/Pavement_Scanning_Cam_Array/MIP_SDK_version/verify_tools/build/_deps/mip_sdk-subbuild/mip_sdk-populate-prefix/src/mip_sdk-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
