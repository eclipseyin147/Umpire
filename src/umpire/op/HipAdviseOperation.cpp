//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/op/HipAdviseOperation.hpp"

#include <sstream>

#include "umpire/util/Macros.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace op {

HipAdviseOperation::HipAdviseOperation(hipMemoryAdvise a) : m_advise(a)
{
}

void HipAdviseOperation::apply(void* src_ptr, util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation), int val,
                               std::size_t length)
{
  int device = val;
  auto error = ::hipMemAdvise(src_ptr, length, m_advise, device);

  if (error != hipSuccess) {
    std::ostringstream oss;
    oss << "hipMemAdvise( src_ptr = " << src_ptr << ", length = " << length
        << ", device = " << device << ") failed with error: " << hipGetErrorString(error);
    UMPIRE_ERROR(runtime_error, oss.str());
  }
}

} // end of namespace op
} // end of namespace umpire
