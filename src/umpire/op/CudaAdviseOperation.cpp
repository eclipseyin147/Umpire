//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/CudaAdviseOperation.hpp"

#include <cuda_runtime_api.h>

#include "umpire/util/error.hpp"

namespace umpire {
namespace op {

CudaAdviseOperation::CudaAdviseOperation(cudaMemoryAdvise a) : m_advice{a}
{
}

void CudaAdviseOperation::apply(void* src_ptr, util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation), int val,
                                std::size_t length)
{
  int device = val;
  cudaError_t error = ::cudaMemAdvise(src_ptr, length, m_advice, device);

  if (error != cudaSuccess) {
    std::ostringstream oss;
    oss << "cudaMemAdvise( src_ptr = {}, length = {}, device = {}) failed with error: {}" << src_ptr<<  length<< device<<cudaGetErrorString(error);
    UMPIRE_ERROR(runtime_error,
                 oss.str());
  }
}

} // end of namespace op
} // end of namespace umpire
