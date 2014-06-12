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

#include <emmintrin.h>
#include <immintrin.h>
#include <cmath>
#include "AvxInternals.hpp"

namespace khyber
{
  namespace avx
  {
    void InternalAdd(size_t size,
                     sp_t* sum,
                     const sp_t* augend,
                     const sp_t* addend)
    {
      __m256* pSum = (__m256*)sum;
      __m256* pAugend = (__m256*)augend;
      __m256* pAddend = (__m256*)addend;
      
      size_t i;
      for ( i = 0; i < (size >> 3); ++i ) {
        pSum[i] = _mm256_add_ps(pAugend[i], pAddend[i]);
      }

      i <<= 3;
      for ( ; i < size; ++i ) {
        sum[i] = augend[i] + addend[i];
      }
    }

    void InternalSub(size_t size,
                     sp_t* difference,
                     const sp_t* minuend,
                     const sp_t* subtrahend)
    {
      __m256* pDifference = (__m256*)difference;
      __m256* pMinuend = (__m256*)minuend;
      __m256* pSubtrahend = (__m256*)subtrahend;

      size_t i;
      for ( i = 0; i < (size >> 3); ++i ) {
        pDifference[i] = _mm256_sub_ps(pMinuend[i], pSubtrahend[i]);
      }

      i <<= 3;
      for ( ; i < size; ++i ) {
        difference[i] = minuend[i] + subtrahend[i];
      }
    }

    void InternalMul(size_t size,
                     sp_t* product,
                     const sp_t* multiplicand,
                     const sp_t* multiplier)
    {
      __m256* pProduct = (__m256*)product;
      __m256* pMultiplicand = (__m256*)multiplicand;
      __m256* pMultiplier = (__m256*)multiplier;

      size_t i;
      for ( i = 0; i < (size >> 3); ++i ) {
        pProduct[i] = _mm256_mul_ps(pMultiplicand[i], pMultiplier[i]);
      }

      i <<= 3;
      for ( ; i < size; ++i ) {
        product[i] = multiplier[i] * multiplicand[i];
      }
    }

    void InternalDiv(size_t size,
                     sp_t* quotient,
                     const sp_t* dividend,
                     const sp_t* divisor)
    {
      __m256* pQuotient = (__m256*)quotient;
      __m256* pDividend = (__m256*)dividend;
      __m256* pDivisor = (__m256*)divisor;

      size_t i;
      for ( i = 0; i < (size >> 3); ++i ) {
        pQuotient[i] = _mm256_div_ps(pDividend[i], pDivisor[i]);
      }

      i <<= 3;
      for ( ; i < size; ++i ) {
        quotient[i] = dividend[i] / divisor[i];
      }
    }

    void InternalSqrt(size_t size,
                      sp_t* dst,
                      sp_t* src)
    {
      __m256* pDst = (__m256*)dst;
      __m256* pSrc = (__m256*)src;

      size_t i;
      for ( i = 0; i < (size >> 3); ++i ) {
        pDst[i] = _mm256_sqrt_ps(pSrc[i]);
      }

      i <<= 3;
      for ( ; i < size; ++i ) {
        dst[i] = sqrt(src[i]);
      }
    }
  }
}
