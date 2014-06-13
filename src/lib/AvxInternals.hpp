// Copyright 2014 Irfan Hamid
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include "Types.hpp"

namespace khyber
{
  namespace avx
  {
    ///
    /// Using Intel64 AVX instructions:
    /// Add the two single-precision floating point arrays addend0 and addend1 of size elements into sum
    /// sum[0:size-1] = addend0[0:size-1] + addend1[0:size-1]
    ///
    void InternalAdd(size_t size,
                     sp_t* sum,
                     const sp_t* augend,
                     const sp_t* addend);

    void InternalSub(size_t size,
                     sp_t* difference,
                     const sp_t* minuend,
                     const sp_t* subtrahend);

    void InternalMul(size_t size,
                     sp_t* product,
                     const sp_t* multiplicand,
                     const sp_t* multiplier);

    void InternalDiv(size_t size,
                     sp_t* quotient,
                     const sp_t* dividend,
                     const sp_t* divisor);

    void InternalDotProduct(size_t size,
                            sp_t* product,
                            const sp_t* multiplier,
                            const sp_t* multiplicand);

    void InternalSqrt(size_t size,
                      sp_t* dst,
                      sp_t* src);
  }
}
