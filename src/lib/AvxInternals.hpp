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
#include "Types.hpp"

namespace khyber
{
  namespace avx
  {
    ///
    /// Using Intel64 AVX instructions:
    /// Add the two single-precision floating point arrays addend0 and addend1 of size elements into sum
    /// sum[0:size-1] = addend0[0:size-1] + addend1[0:size-1]
    ///
    void InternalAdd(size_t size,
                     sp_t* sum,
                     const sp_t* addend0,
                     const sp_t* addend1);
    
    /*
    ///
    /// Using Intel64 AVX instructions:
    /// Add the single-precision floating point array addend of size elements into acc
    /// acc[0:size-1] = acc[0:size-1] + addend[0:size-1]
    ///
    void InternalAddAcc(size_t size,
                        sp_t* acc,
                        const sp_t* addend);*/
  }
}
