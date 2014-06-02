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

#include "Array.hpp"
#include "AvxInternals.hpp"
#include "Avx2Internals.hpp"

namespace khyber
{
  template<>
  Array<sp_t> Array<sp_t>::Add(const Array<sp_t>& addend) const
  {
    // return AddImpl(this, addend);

    // This ugliness (and the one in AddAcc( ) is for pointer-to-member-function
    return (this->*AddImpl)(addend);
  }
  
  template<>
  Array<sp_t>& Array<sp_t>::AddAcc(const Array<sp_t>& addend)
  {
    //return AddAccImpl(this, addend);
    return (this->*AddAccImpl)(addend);
  }
  
  template<>
  Array<sp_t> Array<sp_t>::Avx2AddImpl(const Array<sp_t>& addend) const
  {
    Array<sp_t> sum(_size);
    avx2::InternalAdd(_size,
                      sum._buffer,
                      _buffer,
                      addend._buffer);
    return std::move(sum);
  }
  
  template<>
  Array<sp_t>& Array<sp_t>::Avx2AddAccImpl(const Array<sp_t>& addend)
  {
    avx2::InternalAddAcc(_size,
                         _buffer,
                         addend._buffer);
    return *this;
  }
  
  template<>
  Array<sp_t> Array<sp_t>::AvxAddImpl(const Array<sp_t>& addend) const
  {
    Array<sp_t> sum(_size);
    avx::InternalAdd(_size,
                     sum._buffer,
                     _buffer,
                     addend._buffer);
    return std::move(sum);
  }
  
  template<>
  Array<sp_t>& Array<sp_t>::AvxAddAccImpl(const Array<sp_t>& addend)
  {
    avx::InternalAddAcc(_size,
                        _buffer,
                        addend._buffer);
    return *this;
  }
  
  template<>
  void Array<sp_t>::BuildArchBinding()
  {
    if ( _procCaps.IsAvx2() ) {
      AddImpl = &Array<sp_t>::Avx2AddImpl;
      AddAccImpl = &Array<sp_t>::Avx2AddAccImpl;
    } else if ( _procCaps.IsAvx() ) {
      AddImpl = &Array<sp_t>::AvxAddImpl;
      AddAccImpl = &Array<sp_t>::AvxAddAccImpl;
    } else {
      AddImpl = &Array<sp_t>::BaseAddImpl;
      AddAccImpl = &Array<sp_t>::BaseAddAccImpl;
    }
  }
}
