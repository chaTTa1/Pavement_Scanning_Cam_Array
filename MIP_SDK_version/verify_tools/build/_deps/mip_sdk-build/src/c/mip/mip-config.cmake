set(mip_VERSION 4.0.0)


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was mip-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

set_and_check(MIP_LIBRARY_DIR "${PACKAGE_PREFIX_DIR}/lib")

set(MIP_INCLUDE_DIRS "")

# Include directories based on supported compilers
if(CMAKE_CXX_COMPILER)
  set_and_check(MIP_CPP_INCLUDE_DIR "${PACKAGE_PREFIX_DIR}/include/microstrain/cpp")
  list(APPEND MIP_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/include/microstrain/cpp")
endif()

if(CMAKE_C_COMPILER)
  set_and_check(MIP_C_INCLUDE_DIR "${PACKAGE_PREFIX_DIR}/include/microstrain/c")
  list(APPEND MIP_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/include/microstrain/c")
endif()

set(MIP_LIBRARY mip)

# Add the link directories so CMake can find the library
link_directories("${MIP_LIBRARY_DIR}")

set(MIP_LIBRARIES "${MIP_LIBRARY}")

# Find components
macro(mipcfg_find_component comp required)
  set(_mip_REQUIRED)
  if(${required} AND mip_FIND_REQUIRED)
    set(_mip_REQUIRED REQUIRED)
  endif()

  set(__mip_comp_nv "${comp}")

  find_package(${__mip_comp_nv} ${mip_VERSION} EXACT ${_mip_REQUIRED} CONFIG)

  set(__mip_comp_found ${${__mip_comp_nv}_FOUND})

  # FindPackageHandleStandardArgs expects <package>_<component>_FOUND
  set(mip_${comp}_FOUND ${__mip_comp_found})

  string(TOUPPER ${comp} _MIP_COMP)
  set(mip_${_MIP_COMP}_FOUND ${__mip_comp_found})

  # Create list of libraries including all the found components
  if(__mip_comp_found)
    list(APPEND MIP_LIBRARIES ${__mip_comp_nv})
  endif()

  unset(_mip_REQUIRED)
  unset(_mip_QUIET)
  unset(__mip_comp_nv)
  unset(__mip_comp_found)
  unset(_MIP_COMP)
endmacro()

# Iterate requested components to find them
foreach(__mip_comp IN LISTS mip_FIND_COMPONENTS)
  mipcfg_find_component(${__mip_comp} ${mip_FIND_REQUIRED_${__mip_comp}} 0)
endforeach()

check_required_components(mip)
