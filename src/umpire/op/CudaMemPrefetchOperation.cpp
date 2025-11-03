//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/CudaMemPrefetchOperation.hpp"

#include <cuda_runtime_api.h>

#include <sstream>

#include "umpire/util/Macros.hpp"
#include "umpire/util/Platform.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace op {

void CudaMemPrefetchOperation::apply(void* src_ptr, util::AllocationRecord* UMPIRE_UNUSED_ARG(allocation), int value,
                                     std::size_t length)
{
  int device{value};
  cudaError_t error;

  // Use current device for properties if device is CPU
  int current_device;
  error = cudaGetDevice(&current_device);
  if (error != cudaSuccess) {
    std::ostringstream oss;
    oss << "cudaGetDevice failed with error: " << cudaGetErrorString(error);
    UMPIRE_ERROR(runtime_error, oss.str());
  }
  int gpu = (device != cudaCpuDeviceId) ? device : current_device;

  cudaDeviceProp properties;
  error = ::cudaGetDeviceProperties(&properties, gpu);
  if (error != cudaSuccess) {
    std::ostringstream oss;
    oss << "cudaGetDeviceProperties( device = " << gpu << " ) failed with error: " << cudaGetErrorString(error);
    UMPIRE_ERROR(runtime_error, oss.str());
  }

  if (properties.managedMemory == 1 && properties.concurrentManagedAccess == 1) {
    error = ::cudaMemPrefetchAsync(src_ptr, length, device);

    if (error != cudaSuccess) {
      std::ostringstream oss;
      oss << "cudaMemPrefetchAsync( src_ptr = " << src_ptr << ", length = " << length
          << ", device = " << device << ") failed with error: " << cudaGetErrorString(error);
      UMPIRE_ERROR(runtime_error, oss.str());
    }
  }
}

camp::resources::EventProxy<camp::resources::Resource> CudaMemPrefetchOperation::apply_async(
    void* src_ptr, util::AllocationRecord* UMPIRE_UNUSED_ARG(allocation), int value, std::size_t length,
    camp::resources::Resource& ctx)
{
  int device{value};
  cudaError_t error;

  // Use current device for properties if device is CPU
  int current_device;
  error = cudaGetDevice(&current_device);
  if (error != cudaSuccess) {
    std::ostringstream oss;
    oss << "cudaGetDevice failed with error: " << cudaGetErrorString(error);
    UMPIRE_ERROR(runtime_error, oss.str());
  }
  int gpu = (device != cudaCpuDeviceId) ? device : current_device;

  cudaDeviceProp properties;
  error = ::cudaGetDeviceProperties(&properties, gpu);
  if (error != cudaSuccess) {
    std::ostringstream oss;
    oss << "cudaGetDeviceProperties( device = " << gpu << " ) failed with error: " << cudaGetErrorString(error);
    UMPIRE_ERROR(runtime_error, oss.str());
  }

  auto resource = ctx.try_get<camp::resources::Cuda>();
  if (!resource) {
    std::ostringstream oss;
    oss << "Expected resources::Cuda, got resources::" << platform_to_string(ctx.get_platform());
    UMPIRE_ERROR(resource_error, oss.str());
  }
  auto stream = resource->get_stream();

  if (properties.managedMemory == 1 && properties.concurrentManagedAccess == 1) {
    error = ::cudaMemPrefetchAsync(src_ptr, length, device, stream);

    if (error != cudaSuccess) {
      std::ostringstream oss;
      oss << "cudaMemPrefetchAsync( src_ptr = " << src_ptr << ", length = " << length
          << ", device = " << device << ", stream = " << (void*)stream
          << ") failed with error: " << cudaGetErrorString(error);
      UMPIRE_ERROR(runtime_error, oss.str());
    }
  }

  return camp::resources::EventProxy<camp::resources::Resource>{ctx};
}

} // end of namespace op
} // end of namespace umpire
