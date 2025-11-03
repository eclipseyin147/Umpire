//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_HipPinnedAllocator_HPP
#define UMPIRE_HipPinnedAllocator_HPP

#include <sstream>

#include <hip/hip_runtime.h>

#include "umpire/alloc/HipAllocator.hpp"
#include "umpire/config.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/Platform.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace alloc {

struct HipPinnedAllocator : HipAllocator {
  using HipAllocator::HipAllocator;

  void* allocate(std::size_t bytes)
  {
    hipError_t error;
    void* ptr{nullptr};

    switch (m_granularity) {
      default:
      case MemoryResourceTraits::granularity_type::unknown:
        UMPIRE_LOG(Debug, "::hipHostMalloc(" << bytes << ")");
        error = ::hipHostMalloc(&ptr, bytes);
        break;

      case MemoryResourceTraits::granularity_type::fine_grained:
#ifdef UMPIRE_ENABLE_HIP_COHERENCE_GRANULARITY
        UMPIRE_LOG(Debug, "::hipHostMalloc(" << bytes << ", hipHostMallocDefault)");
        error = ::hipHostMalloc(&ptr, bytes, hipHostMallocDefault);
#else
        {
          std::ostringstream oss;
          oss << "Fine grained memory coherence not supported for allocation";
          UMPIRE_ERROR(runtime_error, oss.str());
        }
#endif // UMPIRE_ENABLE_HIP_COHERENCE_GRANULARITY
        break;

      case MemoryResourceTraits::granularity_type::coarse_grained:
#ifdef UMPIRE_ENABLE_HIP_COHERENCE_GRANULARITY
        UMPIRE_LOG(Debug, "::hipHostMalloc(" << bytes << ", hipHostMallocNonCoherent)");
        error = ::hipHostMalloc(&ptr, bytes, hipHostMallocNonCoherent);
#else
        {
          std::ostringstream oss;
          oss << "Coarse grained memory coherence not supported for allocation";
          UMPIRE_ERROR(runtime_error, oss.str());
        }
#endif // UMPIRE_ENABLE_HIP_COHERENCE_GRANULARITY
        break;
    }

    UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);
    if (error != hipSuccess) {
      if (error == hipErrorMemoryAllocation) {
        std::ostringstream oss;
        oss << "hipHostMalloc( bytes = " << bytes << " ) failed with error: " << hipGetErrorString(error);
        UMPIRE_ERROR(out_of_memory_error, oss.str());
      } else {
        std::ostringstream oss;
        oss << "hipHostMalloc( bytes = " << bytes << " ) failed with error: " << hipGetErrorString(error);
        UMPIRE_ERROR(runtime_error, oss.str());
      }
    }

    return ptr;
  }

  void deallocate(void* ptr)
  {
    UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
    hipError_t error = ::hipHostFree(ptr);
    if (error != hipSuccess) {
      std::ostringstream oss;
      oss << "hipHostFree( ptr = " << ptr << " ) failed with error: " << hipGetErrorString(error);
      UMPIRE_ERROR(runtime_error, oss.str());
    }
  }

  bool isAccessible(Platform p)
  {
    if (p == Platform::hip || p == Platform::host)
      return true;
    else
      return false; // p is undefined
  }
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_HipPinnedAllocator_HPP
