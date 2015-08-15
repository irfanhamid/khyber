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

#include "Enable.hpp"
#include <immintrin.h>
#include <cmath>
#include "Avx2Internals.hpp"

namespace khyber
{
  namespace avx2
  {
    void InternalAdd(size_t size,
                     sp_t* sum,
                     sp_t* augend,
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

    void InternalNegate(size_t size,
                        sp_t *dst,
                        sp_t *src)
    {
      __m256i* pDst = (__m256i*)dst;
      __m256i* pSrc = (__m256i*)src;
      __m256i mask = _mm256_set_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);

      size_t i;
      for ( i = 0; i < (size >> 3); ++i ) {
        pDst[i] = _mm256_xor_si256(pSrc[i], mask);
      }

      i <<= 3;
      for ( ; i < size; ++i ) {
        dst[i] = -src[i];
      }
    }
  }
}
