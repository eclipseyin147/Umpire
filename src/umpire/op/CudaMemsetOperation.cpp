//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/CudaMemsetOperation.hpp"

#include <cuda_runtime_api.h>

#include "umpire/util/Macros.hpp"
#include "umpire/util/Platform.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace op {

void CudaMemsetOperation::apply(void* src_ptr, util::AllocationRecord* UMPIRE_UNUSED_ARG(allocation), int value,
                                std::size_t length)
{
  cudaError_t error = ::cudaMemset(src_ptr, value, length);

  if (error != cudaSuccess) {
    std::ostringstream oss;
    oss << "cudaMemset( src_ptr = {}, val = {}, length = {}) failed with error: {}" <<  src_ptr<<value<< length<< cudaGetErrorString(error);
    UMPIRE_ERROR(runtime_error, oss.str());
  }
}

camp::resources::EventProxy<camp::resources::Resource> CudaMemsetOperation::apply_async(
    void* src_ptr, util::AllocationRecord* UMPIRE_UNUSED_ARG(allocation), int value, std::size_t length,
    camp::resources::Resource& ctx)
{
  auto device = ctx.try_get<camp::resources::Cuda>();
  if (!device) {
    std::ostringstream oss;
    oss << "Expected resources::Cuda, got resources::{}" <<  platform_to_string(ctx.get_platform());
    UMPIRE_ERROR(resource_error,
                 oss.str());
  }
  auto stream = device->get_stream();

  cudaError_t error = ::cudaMemsetAsync(src_ptr, value, length, stream);

  if (error != cudaSuccess) {
    std::ostringstream oss;
    oss << "cudaMemsetAsync( src_ptr = {}, value = {}, length = {}, stream = {}) failed with error: {}" << src_ptr<< value<< length<<cudaGetErrorString(error)<< (void*)stream;
    UMPIRE_ERROR(
        runtime_error,
        oss.str());
  }

  return camp::resources::EventProxy<camp::resources::Resource>{ctx};
}

} // end of namespace op
} // end of namespace umpire
