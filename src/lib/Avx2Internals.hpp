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

namespace khyber
{
  ///
  /// \brief Low-level functions for vector operations using the AVX2 instruction set.
  ///
  /// This namespace provides C-style functions for computation using the AVX2 instruction set. These functions are used internally by
  /// \link Array<T>\endlink to carry out the actual computations when the \link ProcessorCaps\endlink indicates that the AVX2 instruction
  /// set is present. While it is possible to use these functions for end-use, it is strongly suggested to use the \link Array<T>\endlink
  /// class instead; the functions in this namespace are not processor aware, if you use them on a processor without AVX2 support, it will
  /// cause a fault, i.e., hardware exception. Additionally, the \link Array<T>\endlink provides growable buffers as well as interface
  /// compatibility between it and the std::vector<T> type.
  ///
  namespace avx2
  {
    ///
    /// \brief InternalAdd add the two float arrays augend and addend together and store the result in sum
    /// \param size the number of elements in all three array parameters
    /// \param sum
    /// \param augend
    /// \param addend
    ///
    void InternalAdd(size_t size,
                     float* sum,
                     float* augend,
                     const float* addend);
    ///
    /// \brief Change the sign of every element in src and store the result in dst
    /// \param size the number of elements in both array parameters
    /// \param dst
    /// \param src
    ///
    void InternalNegate(size_t size,
                        float* dst,
                        float* src);
  }
}
