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
                     const sp_t* addend0,
                     const sp_t* addend1)
    {
      __m256* pSum = (__m256*)sum;
      __m256* pAddend0 = (__m256*)addend0;
      __m256* pAddend1 = (__m256*)addend1;
      
      for ( size_t i = 0; i < (size >> 3); ++i ) {
        pSum[i] = _mm256_add_ps(pAddend0[i], pAddend1[i]);
      }
    }

    void InternalSub(size_t size,
                     sp_t* difference,
                     const sp_t* minuend,
                     const sp_t* subtrahend)
    {
    }

    void InternalMul(size_t size,
                     sp_t* product,
                     const sp_t* multiplicand,
                     const sp_t* multiplier)
    {
    }

    void InternalDiv(size_t size,
                     sp_t* quotient,
                     const sp_t* dividend,
                     const sp_t* divisor)
    {
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