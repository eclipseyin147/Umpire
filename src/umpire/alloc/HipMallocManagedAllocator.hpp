//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_HipMallocManagedAllocator_HPP
#define UMPIRE_HipMallocManagedAllocator_HPP

#include <sstream>

#include "hip/hip_runtime_api.h"
#include "umpire/alloc/HipAllocator.hpp"
#include "umpire/config.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/Platform.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace alloc {

/*!
 * \brief Uses hipMallocManaged and hipFree to allocate and deallocate
 *        unified memory on AMD GPUs.
 */
struct HipMallocManagedAllocator : HipAllocator {
  using HipAllocator::HipAllocator;

  /*!
   * \brief Allocate bytes of memory using hipMallocManaged.
   *
   * \param bytes Number of bytes to allocate.
   *
   * \return Pointer to start of the allocation.
   *
   * \throws umpire::util::runtime_error if memory cannot be allocated.
   */
  void* allocate(std::size_t bytes)
  {
    void* ptr{nullptr};

    hipError_t error = ::hipMallocManaged(&ptr, bytes);
    UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);
    if (error != hipSuccess) {
      if (error == hipErrorMemoryAllocation) {
        std::ostringstream oss;
        oss << "hipMallocManaged( bytes = " << bytes << " ) failed with error: " << hipGetErrorString(error);
        UMPIRE_ERROR(out_of_memory_error, oss.str());
      } else {
        std::ostringstream oss;
        oss << "hipMallocManaged( bytes = " << bytes << " ) failed with error: " << hipGetErrorString(error);
        UMPIRE_ERROR(runtime_error, oss.str());
      }
    }

#ifdef UMPIRE_ENABLE_HIP_COHERENCE_GRANULARITY
    if (m_granularity == MemoryResourceTraits::granularity_type::coarse_grained) {
      int device;

      hipError_t error = ::hipGetDevice(&device);
      if (error != hipSuccess) {
        std::ostringstream oss;
        oss << "hipGetDevice failed with error: " << hipGetErrorString(error);
        UMPIRE_ERROR(runtime_error, oss.str());
      }

      UMPIRE_LOG(Debug, "::hipMemAdvise(hipMemAdviseSetCoarseGrain)");
      error = ::hipMemAdvise(ptr, bytes, hipMemAdviseSetCoarseGrain, device);

      if (error != hipSuccess) {
        std::ostringstream oss;
        oss << "hipMemAdvise( src_ptr = " << ptr << ", length = " << bytes << ", device = " << device << ") failed with error: " << hipGetErrorString(error);
        UMPIRE_ERROR(runtime_error, oss.str());
      }
    }
#endif // UMPIRE_ENABLE_HIP_COHERENCE_GRANULARITY

    return ptr;
  }

  /*!
   * \brief Deallocate memory using hipFree.
   *
   * \param ptr Address to deallocate.
   *
   * \throws umpire::util::runtime_error if memory be free'd.
   */
  void deallocate(void* ptr)
  {
    UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");

    hipError_t error = ::hipFree(ptr);
    if (error != hipSuccess) {
      std::ostringstream oss;
      oss << "hipFree( ptr = " << ptr << " ) failed with error: " << hipGetErrorString(error);
      UMPIRE_ERROR(runtime_error, oss.str());
    }
  }

  bool isAccessible(Platform p)
  {
    if (p == Platform::hip || p == Platform::host)
      return true;
    else
      return false;
  }
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_hipMallocManagedAllocator_HPP
