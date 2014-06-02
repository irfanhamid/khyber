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
                     sp_t* addend0,
                     sp_t* addend1)
    {
      __m256* pSum = (__m256*)sum;
      __m256* pAddend0 = (__m256*)addend0;
      __m256* pAddend1 = (__m256*)addend1;

      for ( size_t i = 0; i < (size >> 3); ++i ) {
        pSum[i] = _mm256_add_ps(pAddend0[i], pAddend1[i]);
      }
    }

    void InternalAddAcc(size_t size,
                        sp_t* acc,
                        sp_t* addend)
    {
    }
  }
}
