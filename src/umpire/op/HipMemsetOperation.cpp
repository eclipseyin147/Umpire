//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/HipMemsetOperation.hpp"

#include <hip/hip_runtime.h>

#include <sstream>

#include "umpire/util/Macros.hpp"
#include "umpire/util/Platform.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace op {

void HipMemsetOperation::apply(void* src_ptr, util::AllocationRecord* UMPIRE_UNUSED_ARG(allocation), int value,
                               std::size_t length)
{
  hipError_t error = ::hipMemset(src_ptr, value, length);

  if (error != hipSuccess) {
    std::ostringstream oss;
    oss << "hipMemset( src_ptr = " << src_ptr << ", value = " << value
        << ", length = " << length << ") failed with error: " << hipGetErrorString(error);
    UMPIRE_ERROR(runtime_error, oss.str());
  }
}

camp::resources::EventProxy<camp::resources::Resource> HipMemsetOperation::apply_async(
    void* src_ptr, util::AllocationRecord* UMPIRE_UNUSED_ARG(allocation), int value, std::size_t length,
    camp::resources::Resource& ctx)
{
  auto device = ctx.try_get<camp::resources::Hip>();
  if (!device) {
    std::ostringstream oss;
    oss << "Expected resources::Hip, got resources::" << platform_to_string(ctx.get_platform());
    UMPIRE_ERROR(resource_error, oss.str());
  }
  auto stream = device->get_stream();

  hipError_t error = ::hipMemsetAsync(src_ptr, value, length, stream);

  if (error != hipSuccess) {
    std::ostringstream oss;
    oss << "hipMemsetAsync( src_ptr = " << src_ptr << ", value = " << value
        << ", length = " << length << ", stream = " << (void*)stream
        << ") failed with error: " << hipGetErrorString(error);
    UMPIRE_ERROR(runtime_error, oss.str());
  }

  return camp::resources::EventProxy<camp::resources::Resource>{ctx};
}

} // end of namespace op
} // end of namespace umpire
