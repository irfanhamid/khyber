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
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T value_type;

    template<typename U>
    struct rebind {
      typedef SimdAllocator<U, alignment> other;
    };

    SimdAllocator() throw() {}

    SimdAllocator(const SimdAllocator&) throw() {}

    template<typename U>
    SimdAllocator(const SimdAllocator<U, alignment>&) throw() {}

    template<typename U>
    SimdAllocator& operator = (const SimdAllocator<U, alignment>&)
    {
      return *this;
    }

    SimdAllocator<T, alignment>& operator = (const SimdAllocator&)
    {
      return *this;
    }

    T* allocate(size_t n)
    {
      return (T*)_mm_malloc(n * sizeof(T), alignment);
    }

    void deallocate(T* p, size_t n)
    {
      _mm_free(p);
    }
  };

  template<typename T, typename U, size_t alignment>
  inline bool operator == (const SimdAllocator<T, alignment>&, const SimdAllocator<U, alignment>&)
  {
    return true;
  }

  template<typename T, typename U, size_t alignment>
  inline bool operator != (const SimdAllocator<T, alignment>& lhs, const SimdAllocator<U, alignment>& rhs)
  {
    return !(lhs == rhs);
  }
}
