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

#include <iostream>
#include <string>
#include <sstream>
#include <boost/cstdint.hpp>
  
#define cpuid(func, eax, ebx, ecx, edx)					\
__asm__ __volatile__ ("cpuid":					\
"=a" (eax), "=b" (ebx), "=c" (ecx), "=d" (edx) : "a" (func));

#define BM_NO 0x00000000
#define BM_00 0x00000001
#define BM_01 0x00000002
#define BM_02 0x00000004
#define BM_03 0x00000008
#define BM_04 0x00000010
#define BM_05 0x00000020
#define BM_06 0x00000040
#define BM_07 0x00000080
#define BM_08 0x00000100
#define BM_09 0x00000200
#define BM_10 0x00000400
#define BM_11 0x00000800
#define BM_12 0x00001000
#define BM_13 0x00002000
#define BM_14 0x00004000
#define BM_15 0x00008000
#define BM_16 0x00010000
#define BM_17 0x00020000
#define BM_18 0x00040000
#define BM_19 0x00080000
#define BM_20 0x00100000
#define BM_21 0x00200000
#define BM_22 0x00400000
#define BM_23 0x00800000
#define BM_24 0x01000000
#define BM_25 0x02000000
#define BM_26 0x04000000
#define BM_27 0x08000000
#define BM_28 0x10000000
#define BM_29 0x20000000
#define BM_30 0x40000000
#define BM_31 0x80000000

#define BM_MMX    BM_00
#define BM_SSE    BM_01
#define BM_SSE2   BM_02
#define BM_SSE3   BM_03
#define BM_SSE4_1 BM_04
#define BM_SSE4_2 BM_05
#define BM_AVX    BM_06
#define BM_AVX2   BM_07
#define BM_FMA    BM_08
#define BM_HTT    BM_09

#define capset(flag, reg, input_mask, output_mask) flag |= ((reg & BM_##input_mask) ? BM_##output_mask : BM_NO)

namespace khyber
{
  class ProcessorCaps
  {
  public:
    ProcessorCaps()
      : HighestFunction(0),
        HighestExtFunction(0),
        _flags(0)
    {
      uint32_t eax;
      uint32_t ebx;
      uint32_t ecx;
      uint32_t edx;

      cpuid(0, eax, ebx, ecx, edx);
      HighestFunction = eax;
      
      cpuid(0x8000, eax, ebx, ecx, edx);
      HighestExtFunction = eax;
      
      if ( HighestFunction >= 1 ) {
        cpuid(1, eax, ebx, ecx, edx);
        
        // Query everything visible under function code 1
        capset(_flags, edx, 23, MMX);
        capset(_flags, edx, 25, SSE);
        capset(_flags, edx, 26, SSE2);
        capset(_flags, edx, 28, HTT);
        capset(_flags, edx, 00, SSE3);
        capset(_flags, ecx, 19, SSE4_1);
        capset(_flags, ecx, 20, SSE4_2);
        capset(_flags, ecx, 28, AVX);
        capset(_flags, ecx, 12, FMA);
      }
      
      if ( HighestFunction >= 7 ) {
        cpuid(7, eax, ebx, ecx, edx);
        capset(_flags, ebx, 05, AVX2);
      }
    }
    
    /**
     * Returns true if the processor supports Hyperthreading Technology
     */
    inline bool IsHtt() const
    {
      return _flags & BM_HTT;
    }
    
    inline bool IsMmx() const
    {
      return _flags & BM_MMX;
    }
    
    inline bool IsSse() const
    {
      return _flags & BM_SSE;
    }
    
    inline bool IsSse2() const
    {
      return _flags & BM_SSE2;
    }
    
    inline bool IsSse3() const
    {
      return _flags & BM_SSE3;
    }
    
    inline bool IsSse4_1() const
    {
      return _flags & BM_SSE4_1;
    }
    
    inline bool IsSse4_2() const
    {
      return _flags & BM_SSE4_2;
    }
    
    inline bool IsAvx() const
    {
      return _flags & BM_AVX;
    }
    
    inline bool IsAvx2() const
    {
      return _flags & BM_AVX2;
    }
    
    inline bool IsFma() const
    {
      return _flags & BM_FMA;
    }
    
    inline std::string GetCapsDescription() const
    {
      std::ostringstream caps_stream;
      caps_stream << "HTT\t" << (IsHtt() ? "yes" : "no") << std::endl;
      caps_stream << "MMX\t" << (IsMmx() ? "yes" : "no") << std::endl;
      caps_stream << "SSE\t" << (IsSse() ? "yes" : "no") << std::endl;
      caps_stream << "SSE2\t" << (IsSse2() ? "yes" : "no") << std::endl;
      caps_stream << "SSE3\t" << (IsSse3() ? "yes" : "no") << std::endl;
      caps_stream << "SSE4.1\t" << (IsSse4_1() ? "yes" : "no") << std::endl;
      caps_stream << "SSE4.2\t" << (IsSse4_2() ? "yes" : "no") << std::endl;
      caps_stream << "AVX\t" << (IsAvx() ? "yes" : "no") << std::endl;
      caps_stream << "AVX2\t" << (IsAvx2() ? "yes" : "no") << std::endl;
      caps_stream << "FMA\t" << (IsFma() ? "yes" : "no") << std::endl;
      
      return caps_stream.str();
    }
    
    std::size_t L1Cacheline;
    std::size_t L2Cacheline;
    uint32_t HighestFunction;
    uint32_t HighestExtFunction;
    
  private:

    // Flag register layout:
    // --------------------------------------------------------------------
    // HTT | FMA | AVX2 | AVX | SSE4.2 | SSE4.1 | SSE3 | SSE2 | SSE | MMX |
    // --------------------------------------------------------------------
    uint64_t _flags;
  };
}