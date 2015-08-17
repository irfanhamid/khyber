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

#include <immintrin.h>
#include <cmath>
#include "AvxInternals.hpp"

namespace khyber
{
  namespace avx
  {
    void InternalAdd(size_t size,
                     float* sum,
                     float* augend,
                     const float* addend)
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
                     float* difference,
                     float* minuend,
                     const float* subtrahend)
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
        difference[i] = minuend[i] - subtrahend[i];
      }
    }

    void InternalMul(size_t size,
                     float* product,
                     float* multiplier,
                     const float* multiplicand)
    {
      __m256* pProduct = (__m256*)product;
      __m256* pMultiplier = (__m256*)multiplier;
      __m256* pMultiplicand = (__m256*)multiplicand;

      size_t i;
      for ( i = 0; i < (size >> 3); ++i ) {
        pProduct[i] = _mm256_mul_ps(pMultiplier[i], pMultiplicand[i]);
      }

      i <<= 3;
      for ( ; i < size; ++i ) {
        product[i] = multiplier[i] * multiplicand[i];
      }
    }

    void InternalScalarMul(size_t size,
                           float multiplier,
                           float *product,
                           float *src)
    {
      __m256* pSrc = (__m256*)src;
      __m256* pProduct = (__m256*)product;
      __m256 ymmMultiplier = _mm256_set1_ps(multiplier);

      size_t i;
      for ( i = 0; i < (size >> 3); ++i ) {
        pProduct[i] = _mm256_mul_ps(pSrc[i], ymmMultiplier);
      }

      i <<= 3;
      for ( ; i < size; ++i ) {
        product[i] = src[i] * multiplier;
      }
    }
    
    void InternalDiv(size_t size,
                     float* quotient,
                     float* dividend,
                     const float* divisor)
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

    void InternalScalarDiv(size_t size,
                           float divisor,
                           float* quotient,
                           float* src)
    {
      __m256* pSrc = (__m256*)src;
      __m256* pQuotient = (__m256*)quotient;
      __m256 ymmDivisor = _mm256_set1_ps(divisor);

      size_t i;
      for ( i = 0; i < (size >> 3); ++i ) {
        pQuotient[i] = _mm256_div_ps(pSrc[i], ymmDivisor);
      }

      i <<= 3;
      for ( ; i < size; ++i ) {
        quotient[i] = src[i] / divisor;
      }
    }

    void InternalSqrt(size_t size,
                      float* dst,
                      float* src)
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

    void InternalSquare(size_t size,
                        float *dst,
                        float *src)
    {
      __m256* pDst = (__m256*)dst;
      __m256* pSrc = (__m256*)src;

      size_t i;
      for ( i = 0; i < (size >> 3); ++i ) {
        pDst[i] = _mm256_mul_ps(pSrc[i], pSrc[i]);
      }

      i <<= 3;
      for ( ; i < size; ++i ) {
        dst[i] = src[i] * src[i];
      }
    }

    void InternalCube(size_t size,
                      float *dst,
                      float *src)
    {
      __m256* pDst = (__m256*)dst;
      __m256* pSrc = (__m256*)src;

      size_t i;
      for ( i = 0; i < (size >> 3); ++i ) {
        pDst[i] = _mm256_mul_ps(pSrc[i], pSrc[i]);
        pDst[i] = _mm256_mul_ps(pDst[i], pSrc[i]);
      }

      i <<= 3;
      for ( ; i < size; ++i ) {
        dst[i] = src[i] * src[i] * src[i];
      }
    }

    void InternalSummation(size_t size,
                           float* sum,
                           const float* src)
    {
      __m256 scratch;
      __m256 accumulator = _mm256_setzero_ps();
      const __m256* pSrc = (const __m256*)src;

      size_t i;
      for ( i = 0; i < ((size >> 3) - 1); i += 2 ) {
        scratch = _mm256_hadd_ps(pSrc[i], pSrc[i + 1]);
        accumulator = _mm256_add_ps(accumulator, scratch);
      }

      float* tmp = (float*)&accumulator;
      *sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
      i <<= 3;
      while ( i < size ) {
        *sum += src[i++];
      }
    }

    void InternalPrefixSum(size_t size, float *dst, float *src)
    {
      // TODO: Implement this.
    }

    void InternalDotProduct(size_t size,
                            float *product,
                            const float *multiplier,
                            const float *multiplicand)
    {
      __m256* pMultiplier = (__m256*)multiplier;
      __m256* pMultiplicand = (__m256*)multiplicand;
      __m256 accumulator = _mm256_setzero_ps();
      __m256 scratch;

      size_t i;
      for ( i = 0; i < (size >> 3); ++i ) {
        scratch = _mm256_mul_ps(pMultiplier[i], pMultiplicand[i]);
        accumulator = _mm256_add_ps(accumulator, scratch);
      }

      float* tmp = (float*)&accumulator;
      *product = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
      i <<= 3;
      for ( ; i < size; ++i ) {
        *product += (multiplier[i] * multiplicand[i]);
      }
    }

    void InternalDotProductFma(size_t size,
                               float* product,
                               const float* multiplier,
                               const float* multiplicand)
    {
      __m256* pMultiplier = (__m256*)multiplier;
      __m256* pMultiplicand = (__m256*)multiplicand;
      __m256 accumulator = _mm256_setzero_ps();

      size_t i;
      for ( i = 0; i < (size >> 3); ++i ) {
        accumulator = _mm256_fmadd_ps(pMultiplier[i], pMultiplicand[i], accumulator);
      }

      float* tmp = (float*)&accumulator;
      *product = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
      i <<= 3;
      for ( ; i < size; ++i ) {
        *product += (multiplier[i] * multiplicand[i]);
      }
    }

    void InternalNegate(size_t size,
                        float *dst,
                        float *src)
    {
      // This implementation uses 128bit XMM registers instead of the 256bit YMM registers because
      // AVX includes only 128bit integer instructions only, unlike for floating point where it can
      // work with 256bit registers.
      __m128i* pDst = (__m128i*)dst;
      __m128i* pSrc = (__m128i*)src;
      __m128i mask = _mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000);

      size_t i;
      for ( i = 0; i < (size >> 2); ++i ) {
        pDst[i] = _mm_xor_si128(pSrc[i], mask);
      }

      i <<= 2;
      for ( ; i < size; ++i ) {
        dst[i] = -src[i];
      }
    }

    void InternalDistance(size_t size,
                          float* distance,
                          const float *v1,
                          const float *v2)
    {
      const __m256* pV1 = (const __m256*)v1;
      const __m256* pV2 = (const __m256*)v2;
      __m256 scratch;
      __m256 accumulator = _mm256_setzero_ps();

      size_t i;
      for ( i = 0; i < (size >> 3); ++i ) {
        scratch = _mm256_sub_ps(pV1[i], pV2[i]);
        scratch = _mm256_mul_ps(scratch, scratch);
        accumulator = _mm256_add_ps(accumulator, scratch);
      }

      float* tmp = (float*)&accumulator;
      *distance = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
      i <<= 3;
      for ( ; i < size; ++i ) {
        *distance += ((v1[i] - v2[i]) * (v1[i] - v2[i]));
      }
      *distance = sqrt(*distance);
    }

    void InternalReciprocate(size_t size,
                             float *dst,
                             float *src)
    {
      __m256* pDst = (__m256*)dst;
      __m256* pSrc = (__m256*)src;

      size_t i;
      for ( i = 0; i < (size >> 3); ++i ) {
        pDst[i] = _mm256_rcp_ps(pSrc[i]);
      }

      i <<= 3;
      for ( ; i < size; ++i ) {
        dst[i] = 1.0 / src[i];
      }
    }
  }
}
