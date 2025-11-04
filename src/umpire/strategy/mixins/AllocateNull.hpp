//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_Zeroer_HPP
#define UMPIRE_Zeroer_HPP

#include "umpire/config.hpp"

namespace umpire {
namespace strategy {

class FixedPool;

namespace mixins {

class UMPIRE_EXPORT AllocateNull
{
  public:
    AllocateNull();

    void* allocateNull();
    bool deallocateNull(void* ptr);

  private:
    FixedPool* m_zero_byte_pool;

};

} // end of namespace mixins
} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_Zeroer_HPP
