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

#include <boost/cstdint.hpp>

#define DEFAULT_ALIGNMENT 32

namespace khyber
{
  typedef uint16_t ui16_t alignas(DEFAULT_ALIGNMENT);
  typedef uint32_t ui32_t alignas(DEFAULT_ALIGNMENT);
  typedef uint64_t ui64_t alignas(DEFAULT_ALIGNMENT);
  typedef int16_t i16_t alignas(DEFAULT_ALIGNMENT);
  typedef int32_t i32_t alignas(DEFAULT_ALIGNMENT);
  typedef int64_t i64_t alignas(DEFAULT_ALIGNMENT);
  typedef float sp_t alignas(DEFAULT_ALIGNMENT);
  typedef double dp_t alignas(DEFAULT_ALIGNMENT);
}
