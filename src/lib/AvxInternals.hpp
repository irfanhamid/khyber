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
  ///
  /// \brief Low-level functions for vector operations using the AVX instruction set.
  ///
  /// This namespace provides C-style functions for computation using the AVX instruction set. These functions are used internally by
  /// \link Array<T>\endlink to carry out the actual computations when the \link ProcessorCaps\endlink indicates that the AVX instruction
  /// set is present. While it is possible to use these functions for end-use, it is strongly suggested to use the \link Array<T>\endlink
  /// class instead; the functions in this namespace are not processor aware, if you use them on a processor without AVX support, it will
  /// cause a fault, i.e., hardware exception.
  ///
  namespace avx
  {
    ///
    /// \brief InternalAdd add the two single-precision floating point arrays augend and addend into sum
    /// \param size the number of elements in all three array parameters
    /// \param sum
    /// \param augend
    /// \param addend
    ///
    void InternalAdd(size_t size,
                     sp_t* sum,
                     sp_t* augend,
                     const sp_t* addend);

    ///
    /// \brief InternalSub subtract the single-precision array subtrahend from minuend and store the result in difference
    /// \param size the number of elements in all three array parameters
    /// \param difference
    /// \param minuend
    /// \param subtrahend
    ///
    void InternalSub(size_t size,
                     sp_t* difference,
                     sp_t* minuend,
                     const sp_t* subtrahend);

    ///
    /// \brief InternalMul multiply the single-precision arrays multiplicand and multiplier and store the result in product. This can be considered
    /// the classical cross product of two vectors
    /// \param size the number of elements in all three array parameters
    /// \param product
    /// \param multiplicand
    /// \param multiplier
    ///
    void InternalMul(size_t size,
                     sp_t* product,
                     sp_t* multiplier,
                     const sp_t* multiplicand);

    ///
    /// \brief Multiply a single-precision array by a scalar
    /// \param size the number of elements in the array product and src
    /// \param multiplier the scalar value by which to multiply the src array
    /// \param product the resulting array, can be the same address as src
    /// \param src the array to multiply
    ///
    void InternalScalarMul(size_t size,
                           sp_t multiplier,
                           sp_t* product,
                           sp_t* src);

    ///
    /// \brief InternalDiv divide the single-precision array dividend by divisor and store the results in quotient
    /// \param size the number of elements in all three array parameters
    /// \param quotient
    /// \param dividend
    /// \param divisor
    ///
    void InternalDiv(size_t size,
                     sp_t* quotient,
                     sp_t* dividend,
                     const sp_t* divisor);

    ///
    /// \brief Divide a single-precision array by a scalar
    /// \param size the number of elements in the array quotient and src
    /// \param divisor the scalar value by which to divide the src array
    /// \param quotient the output array, can be the same address as src
    /// \param src the array to divide
    ///
    void InternalScalarDiv(size_t size,
                           sp_t divisor,
                           sp_t* quotient,
                           sp_t* src);

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
    /// \brief InternalSquare compute the square of every element in the array src and store it in the array dst
    /// \param size the number of elements in both array parameters
    /// \param dst
    /// \param src
    ///
    void InternalSquare(size_t size,
                        sp_t* dst,
                        sp_t* src);

    ///
    /// \brief Compute the cube of every element in the array src and store it in the array dst.
    /// \param size the number of elements in both array parameters
    /// \param dst
    /// \param src
    ///
    void InternalCube(size_t size,
                      sp_t* dst,
                      sp_t* src);

    ///
    /// \brief Compute the sum of all elements in the single-precision array src, the output is a scalar. Note this is not the prefix sum,
    /// the src array remains unchanged.
    /// \param size the number of elements in both array parameters
    /// \param sum
    /// \param src
    ///
    void InternalSummation(size_t size,
                           sp_t* sum,
                           const sp_t* src);

    ///
    /// \brief InternalPrefixSum compute the prefix sum (each element is the cumulative sum of all elements up to and including it) of the src array and store it in dst
    /// \param size
    /// \param dst
    /// \param src
    ///
    void InternalPrefixSum(size_t size,
                           sp_t* dst,
                           sp_t* src);

    ///
    /// \brief InternalDotProduct compute the dot product of the single-precision arrays multiplicand and multiplier, store in product
    /// \param size the number of elements in both array parameters
    /// \param product pointer to a scalar single-precision float into which the dot product will be output
    /// \param multiplier
    /// \param multiplicand
    ///
    void InternalDotProduct(size_t size,
                            sp_t* product,
                            const sp_t* multiplier,
                            const sp_t* multiplicand);

    ///
    /// \brief InternalDotProductFma compute the dot product of the single-precision arrays multiplicand and multiplier using the FMA intrinsic, store in product
    /// \param size the number of elements in both array parameters
    /// \param product pointer to a scalar single-precision float into which the dot product will be output
    /// \param multiplier
    /// \param multiplicand
    ///
    void InternalDotProductFma(size_t size,
                               sp_t* product,
                               const sp_t* multiplier,
                               const sp_t* multiplicand);

    ///
    /// \brief Change the sign of every element in src and store the result in dst
    /// \param size the number of elements in both array parameters
    /// \param dst
    /// \param src
    ///
    void InternalNegate(size_t size,
                        sp_t* dst,
                        sp_t* src);

    ///
    /// \brief Compute the linear (geometric) distance between the two vectors, store in distance
    /// \param size the number of elements in both array parameters
    /// \param distance pointer to a scalar single-precision floating point into which the distance will be output
    /// \param v1 array representing 1st point in size()-dimensional space
    /// \param v2 array representing 2nd point in size()-dimensional space
    ///
    void InternalDistance(size_t size,
                          sp_t* distance,
                          const sp_t* v1,
                          const sp_t* v2);

    ///
    /// \brief Compute the reciprocal of every element in the src array and store it in the dst array
    /// \param size the number of elements in both array parameters
    /// \param dst destination array
    /// \param src source array
    ///
    void InternalReciprocate(size_t size,
                             sp_t* dst,
                             sp_t* src);
  }
}
