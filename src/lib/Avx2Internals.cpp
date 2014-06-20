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
#include <avxintrin.h>
#include <avx2intrin.h>
#include <cmath>
#include "Avx2Internals.hpp"

namespace khyber
{
  namespace avx2
  {
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
