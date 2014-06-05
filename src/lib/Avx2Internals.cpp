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
#include <iostream>
#include "Avx2Internals.hpp"

namespace khyber
{
  namespace avx2
  {
    void InternalAdd(size_t size,
                     sp_t* sum,
                     const sp_t* augend,
                     const sp_t* addend)
    {
      __m256* pSum = (__m256*)sum;
      __m256* pAugend = (__m256*)augend;
      __m256* pAddend = (__m256*)addend;

      for ( size_t i = 0; i < (size >> 3); ++i ) {
        pSum[i] = _mm256_add_ps(pAugend[i], pAddend[i]);
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
  }
}
