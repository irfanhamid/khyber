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
  namespace avx2
  {
    ///
    /// \brief InternalAdd add the two single-precision floating point arrays augend and addend into sum using AVX2 instructions
    /// \param size the number of elements in all three array parameters
    /// \param sum
    /// \param augend
    /// \param addend
    ///
    void InternalAdd(size_t size,
                     sp_t* sum,
                     const sp_t* augend,
                     const sp_t* addend);

    ///
    /// \brief InternalSub subtract the single-precision array subtrahend from minuend and store the result in difference using AVX2 instructions
    /// \param size the number of elements in all three array parameters
    /// \param difference
    /// \param minuend
    /// \param subtrahend
    ///
    void InternalSub(size_t size,
                     sp_t* difference,
                     const sp_t* minuend,
                     const sp_t* subtrahend);

    ///
    /// \brief InternalMul multiply the single-precision arrays multiplicand and multiplier and store the result in product. This can be considered the classical cross product of two vectors
    /// \param size the number of elements in all three array parameters
    /// \param product
    /// \param multiplicand
    /// \param multiplier
    ///
    void InternalMul(size_t size,
                     sp_t* product,
                     const sp_t* multiplicand,
                     const sp_t* multiplier);

    ///
    /// \brief InternalDiv
    /// \param size the number of elements in all three array parameters
    /// \param quotient
    /// \param dividend
    /// \param divisor
    ///
    void InternalDiv(size_t size,
                     sp_t* quotient,
                     const sp_t* dividend,
                     const sp_t* divisor);

    ///
    /// \brief InternalSqrt compute the square root of every element in the single-precision array src and store the results in dst
    /// \param size the number of elements in both array parameters
    /// \param dst
    /// \param src
    ///
    void InternalSqrt(size_t size,
                      sp_t* dst,
                      sp_t* src);

    ///
    /// \brief InternalDotProduct compute the dot product of the single-precision arrays multiplicand and multiplier, store in product
    /// \param size the number of elements in both array parameters
    /// \param product pointer to a scalar single-precision parameter
    /// \param multiplicand
    /// \param multiplier
    ///
    void InternalDotProduct(size_t size,
                            sp_t* product,
                            const sp_t* multiplier,
                            const sp_t* multiplicand);
  }
}
