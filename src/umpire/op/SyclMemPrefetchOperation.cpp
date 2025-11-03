//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/SyclMemPrefetchOperation.hpp"

#include <sstream>

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/util/error.hpp"
#include "umpire/util/sycl_compat.hpp"

namespace umpire {
namespace op {

void SyclMemPrefetchOperation::apply(void* src_ptr, util::AllocationRecord* allocation, int value, std::size_t length)
{
  if (allocation->strategy->getTraits().id != value) {
    std::ostringstream oss;
    oss << "SYCL memPrefetch failed with invalid deviceID  = " << value << " ";
    UMPIRE_ERROR(runtime_error, oss.str());
  }

  auto sycl_queue = allocation->strategy->getTraits().queue;

  sycl::usm::alloc src_ptr_kind = get_pointer_type(src_ptr, sycl_queue->get_context());

  if (sycl_queue->get_device().get_info<sycl::info::device::usm_shared_allocations>() &&
      sycl::usm::alloc::shared == src_ptr_kind) {
    sycl_queue->prefetch(src_ptr, length);
    sycl_queue->wait();
  } else {
    std::ostringstream oss;
    oss << "SYCL memPrefetch failed ( bytes = " << length << " )";
    UMPIRE_ERROR(runtime_error, oss.str());
  }
}

} // end of namespace op
} // end of namespace umpire
