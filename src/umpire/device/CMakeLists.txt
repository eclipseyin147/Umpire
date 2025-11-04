##############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

if (UMPIRE_ENABLE_CUDA)
  set_source_files_properties(
    ${PROJECT_SOURCE_DIR}/src/umpire/DeviceAllocator.cpp
    PROPERTIES
    LANGUAGE CUDA)
  set_source_files_properties(
    ${PROJECT_SOURCE_DIR}/src/umpire/device_allocator_helper.cpp
    PROPERTIES
    LANGUAGE CUDA)
  set(umpire_device_depends
    umpire
    camp
    cuda
    umpire_resource
    umpire_alloc
    umpire_op
    umpire_util
    umpire_event
    umpire_strategy)
elseif (UMPIRE_ENABLE_HIP)
  set(umpire_device_depends
    umpire
    camp
    blt::hip
    umpire_resource
    umpire_alloc
    umpire_op
    umpire_util
    umpire_event
    umpire_strategy)
endif()

# Create the device library target
blt_add_library(
  NAME umpire_device
  HEADERS
    ${PROJECT_SOURCE_DIR}/src/umpire/DeviceAllocator.hpp
    ${PROJECT_SOURCE_DIR}/src/umpire/device_allocator_helper.hpp
  SOURCES
    ${PROJECT_SOURCE_DIR}/src/umpire/DeviceAllocator.cpp
    ${PROJECT_SOURCE_DIR}/src/umpire/device_allocator_helper.cpp
  DEPENDS_ON ${umpire_device_depends})

# Fix MSVC runtime conflicts
if (MSVC)
  set_target_properties(umpire_device PROPERTIES
    MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

  target_link_options(umpire_device PRIVATE "/NODEFAULTLIB:LIBCMT")

  # Note: Removed direct object file linking - umpire_device already depends on umpire library
  # which exports all necessary symbols via UMPIRE_EXPORT macros
else()
  set_target_properties(umpire_device PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS "YES")
endif()

# Configure CUDA separable compilation
if (UMPIRE_ENABLE_CUDA)
  set_target_properties(umpire_device PROPERTIES
    CUDA_SEPARABLE_COMPILATION On)
else()
  if (UMPIRE_ENABLE_HIP)
    target_link_options(umpire_device PUBLIC -fgpu-rdc)
    target_compile_options(umpire_device PUBLIC -fgpu-rdc)
  endif()
endif()

target_include_directories(
  umpire_device
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(TARGETS
  umpire_device
  EXPORT umpire-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})