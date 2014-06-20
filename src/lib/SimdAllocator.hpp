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

#include <stdlib.h>
#include <xmmintrin.h>

namespace khyber
{
  ///
  /// \brief Aligned memory allocator for working with SIMD (SSE, AVX) instructions
  ///
  template<typename T, size_t alignment>
  struct SimdAllocator
  {
    typedef T value_type;

    T* allocate(size_t n)
    {
      return (T*)_mm_malloc(n * sizeof(T), alignment);
    }

    void deallocate(T* p, size_t n)
    {
      _mm_free(p);
    }

    bool operator == (const SimdAllocator<T, alignment>& rhs)
    {
      return true;
    }

    bool operator != (const SimdAllocator<T, alignment>& rhs)
    {
      return false;
    }
  };
}
