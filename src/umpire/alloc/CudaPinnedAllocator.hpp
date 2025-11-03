//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_CudaPinnedAllocator_HPP
#define UMPIRE_CudaPinnedAllocator_HPP

#include <cuda_runtime_api.h>

#include <sstream>

#include "umpire/util/Macros.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace alloc {

struct CudaPinnedAllocator {
  void* allocate(std::size_t bytes)
  {
    void* ptr{nullptr};

    cudaError_t error = ::cudaMallocHost(&ptr, bytes);
    UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);
    if (error != cudaSuccess) {
      std::ostringstream oss;
      oss << "cudaMallocHost( bytes = " << bytes << " ) failed with error: " << cudaGetErrorString(error);
      if (error == cudaErrorMemoryAllocation) {
        UMPIRE_ERROR(out_of_memory_error, oss.str());
      } else {
        UMPIRE_ERROR(runtime_error, oss.str());
      }
    }

    return ptr;
  }

  void deallocate(void* ptr)
  {
    UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
    cudaError_t error = ::cudaFreeHost(ptr);
    if (error != cudaSuccess) {
      std::ostringstream oss;
      oss << "cudaFreeHost( ptr = " << ptr << " ) failed with error: " << cudaGetErrorString(error);
      UMPIRE_ERROR(runtime_error, oss.str());
    }
  }

  bool isAccessible(Platform p)
  {
    if (p == Platform::cuda || p == Platform::host)
      return true;
    else
      return false; // p is undefined
  }
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_CudaPinnedAllocator_HPP
