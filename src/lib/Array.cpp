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
    // This ugliness (and the one in AddAcc( ) is for pointer-to-member-function
    // I know we could use boost::function, and I did, but that is much slower
    return (this->*AddImpl)(addend);
  }
  
  template<>
  Array<sp_t>& Array<sp_t>::AddAcc(const Array<sp_t>& addend)
  {
    return (this->*AddAccImpl)(addend);
  }
  
  template<>
  Array<sp_t> Array<sp_t>::Avx2AddImpl(const Array<sp_t>& addend) const
  {
    Array<sp_t> sum(this->_buffer.size());
    avx2::InternalAdd(this->_buffer.size(),
                      sum._buffer.data(),
                      this->_buffer.data(),
                      addend._buffer.data());
    return std::move(sum);
  }
  
  template<>
  Array<sp_t>& Array<sp_t>::Avx2AddAccImpl(const Array<sp_t>& addend)
  {
    avx2::InternalAdd(this->_buffer.size(),
                      this->_buffer.data(),
                      this->_buffer.data(),
                      addend._buffer.data());
    return *this;
  }
  
  template<>
  Array<sp_t> Array<sp_t>::AvxAddImpl(const Array<sp_t>& addend) const
  {
    Array<sp_t> sum(this->_buffer.size());
    avx::InternalAdd(this->_buffer.size(),
                     sum._buffer.data(),
                     this->_buffer.data(),
                     addend._buffer.data());
    return std::move(sum);
  }
  
  template<>
  Array<sp_t>& Array<sp_t>::AvxAddAccImpl(const Array<sp_t>& addend)
  {
    avx::InternalAdd(this->_buffer.size(),
                     this->_buffer.data(),
                     this->_buffer.data(),
                     addend._buffer.data());
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
