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
  /////////////////////////// Programmer API //////////////////////////////////

  template<>
  Array<sp_t> Array<sp_t>::Add(const Array<sp_t>& addend)
  {
    // This ugliness (and the one in AddAcc( ) is for pointer-to-member-function
    // I know we could use boost::function, and I did, but that is much slower
    return (this->*AddImpl)(addend);
  }
  
  template<>
  Array<sp_t>& Array<sp_t>::Add(Array<sp_t> &augend,
                                const Array<sp_t> &addend)
  {
    return (this->*Add2Impl)(augend, addend);
  }

  template<>
  Array<sp_t> Array<sp_t>::Sub(const Array<sp_t>& subtrahend)
  {
    return (this->*SubImpl)(subtrahend);
  }

  template<>
  Array<sp_t>& Array<sp_t>::Sub(Array<sp_t> &minuend,
                                const Array<sp_t> &subtrahend)
  {
    return (this->*Sub2Impl)(minuend, subtrahend);
  }

  template<>
  Array<sp_t> Array<sp_t>::Mul(const Array<sp_t> &multiplier)
  {
    return (this->*MulImpl)(multiplier);
  }

  template<>
  Array<sp_t>& Array<sp_t>::Mul(Array<sp_t> &multiplier,
                                const Array<sp_t> &multiplicand)
  {
    return (this->*Mul2Impl)(multiplier, multiplicand);
  }

  template<>
  Array<sp_t> Array<sp_t>::Div(const Array<sp_t> &divisor)
  {
    return (this->*DivImpl)(divisor);
  }

  template<>
  Array<sp_t>& Array<sp_t>::Div(Array<sp_t> &dividend,
                                const Array<sp_t> &divisor)
  {
    return (this->*Div2Impl)(dividend, divisor);
  }

  template<>
  Array<sp_t> Array<sp_t>::Sqrt()
  {
    return (this->*SqrtImpl)();
  }

  /////////////////////////////////////////////////////////////////////////////
  


  /////////////////////// AVX implementation dispatchers //////////////////////

  template<>
  Array<sp_t> Array<sp_t>::AvxAddImpl(const Array<sp_t>& addend)
  {
    Array<sp_t> sum(this->_buffer.size());
    avx::InternalAdd(this->_buffer.size(),
                     sum._buffer.data(),
                     this->_buffer.data(),
                     addend._buffer.data());
    return std::move(sum);
  }

  template<>
  Array<sp_t>& Array<sp_t>::AvxAdd2Impl(Array<sp_t> &augend,
                                        const Array<sp_t> &addend)
  {
    avx::InternalAdd(this->_buffer.size(),
                     this->_buffer.data(),
                     augend._buffer.data(),
                     addend._buffer.data());
    return *this;
  }
  
  template<>
  Array<sp_t> Array<sp_t>::AvxSubImpl(const Array<sp_t> &subtrahend)
  {
    Array<sp_t> difference(this->_buffer.size());
    avx::InternalSub(this->_buffer.size(),
                     difference._buffer.data(),
                     this->_buffer.data(),
                     subtrahend._buffer.data());
    return std::move(difference);
  }

  template<>
  Array<sp_t>& Array<sp_t>::AvxSub2Impl(Array<sp_t>& minuend,
                                        const Array<sp_t>& subtrahend)
  {
    avx::InternalSub(this->size(),
                     this->data(),
                     minuend.data(),
                     subtrahend.data());
    return *this;
  }

  template<>
  Array<sp_t> Array<sp_t>::AvxMulImpl(const Array<sp_t> &multiplier)
  {
    Array<sp_t> product(this->size());
    avx::InternalMul(this->size(),
                     product.data(),
                     this->data(),
                     multiplier.data());
    return std::move(product);
  }

  template<>
  Array<sp_t>& Array<sp_t>::AvxMul2Impl(Array<sp_t> &multiplier,
                                        const Array<sp_t> &multiplicand)
  {
    avx::InternalMul(this->size(),
                     this->data(),
                     multiplier.data(),
                     multiplicand.data());
    return *this;
  }

  template<>
  Array<sp_t> Array<sp_t>::AvxDivImpl(const Array<sp_t> &divisor)
  {
    Array<sp_t> quotient(divisor.size());
    avx::InternalDiv(this->size(),
                     quotient.data(),
                     this->data(),
                     divisor.data());
    return std::move(quotient);
  }

  template<>
  Array<sp_t>& Array<sp_t>::AvxDiv2Impl(Array<sp_t> &dividend,
                                        const Array<sp_t> &divisor)
  {
    avx::InternalDiv(this->size(),
                     this->data(),
                     dividend.data(),
                     divisor.data());
    return *this;
  }


  template<>
  Array<sp_t> Array<sp_t>::AvxSqrtImpl()
  {
    Array<sp_t> result(this->_buffer.size());
    avx::InternalSqrt(this->_buffer.size(),
                      result._buffer.data(),
                      this->_buffer.data());
    return std::move(result);
  }

  /////////////////////////////////////////////////////////////////////////////



  ////////////////////// AVX2 implementation dispatchers //////////////////////

  template<>
  Array<sp_t> Array<sp_t>::Avx2AddImpl(const Array<sp_t>& addend)
  {
    Array<sp_t> sum(this->_buffer.size());
    avx2::InternalAdd(this->_buffer.size(),
                      sum._buffer.data(),
                      this->_buffer.data(),
                      addend._buffer.data());
    return std::move(sum);
  }

  template<>
  Array<sp_t>& Array<sp_t>::Avx2Add2Impl(Array<sp_t>& augend,
                                         const Array<sp_t>& addend)
  {
    avx2::InternalAdd(this->_buffer.size(),
                      this->_buffer.data(),
                      augend._buffer.data(),
                      addend._buffer.data());
    return *this;
  }

  template<>
  Array<sp_t> Array<sp_t>::Avx2SubImpl(const Array<sp_t> &subtrahend)
  {
    Array<sp_t> difference(this->_buffer.size());
    avx2::InternalSub(this->_buffer.size(),
                      difference._buffer.data(),
                      this->_buffer.data(),
                      subtrahend._buffer.data());
    return std::move(difference);
  }

  template<>
  Array<sp_t>& Array<sp_t>::Avx2Sub2Impl(Array<sp_t>& minuend,
                                         const Array<sp_t>& subtrahend)
  {
    avx2::InternalSub(this->size(),
                      this->data(),
                      minuend.data(),
                      subtrahend.data());
    return *this;
  }

  template<>
  Array<sp_t> Array<sp_t>::Avx2MulImpl(const Array<sp_t> &multiplier)
  {
    Array<sp_t> product(this->size());
    avx2::InternalMul(this->size(),
                      product.data(),
                      this->data(),
                      multiplier.data());
    return std::move(product);
  }

  template<>
  Array<sp_t>& Array<sp_t>::Avx2Mul2Impl(Array<sp_t> &multiplier,
                                         const Array<sp_t> &multiplicand)
  {
    avx2::InternalMul(this->size(),
                      this->data(),
                      multiplier.data(),
                      multiplicand.data());
    return *this;
  }

  template<>
  Array<sp_t> Array<sp_t>::Avx2DivImpl(const Array<sp_t> &divisor)
  {
    Array<sp_t> quotient(divisor.size());
    avx2::InternalDiv(this->size(),
                      quotient.data(),
                      this->data(),
                      divisor.data());
    return std::move(quotient);
  }

  template<>
  Array<sp_t>& Array<sp_t>::Avx2Div2Impl(Array<sp_t> &dividend,
                                         const Array<sp_t> &divisor)
  {
    avx2::InternalDiv(this->size(),
                      this->data(),
                      dividend.data(),
                      divisor.data());
    return *this;
  }

  template<>
  Array<sp_t> Array<sp_t>::Avx2SqrtImpl()
  {
    Array<sp_t> result(this->_buffer.size());
    avx2::InternalSqrt(this->_buffer.size(),
                       result._buffer.data(),
                       this->_buffer.data());
    return std::move(result);
  }

  /////////////////////////////////////////////////////////////////////////////

  template<>
  void Array<sp_t>::BuildFallbackArchBinding()
  {
    AddImpl = &Array<sp_t>::FallbackAddImpl;
    Add2Impl = &Array<sp_t>::FallbackAdd2Impl;
    SubImpl = &Array<sp_t>::FallbackSubImpl;
    Sub2Impl = &Array<sp_t>::FallbackSub2Impl;
    MulImpl = &Array<sp_t>::FallbackMulImpl;
    Mul2Impl = &Array<sp_t>::FallbackMul2Impl;
    DivImpl = &Array<sp_t>::FallbackDivImpl;
    Div2Impl = &Array<sp_t>::FallbackDiv2Impl;
    SqrtImpl = &Array<sp_t>::FallbackSqrtImpl;
  }

  template<>
  void Array<sp_t>::BuildAvxArchBinding()
  {
    AddImpl = &Array<sp_t>::AvxAddImpl;
    Add2Impl = &Array<sp_t>::AvxAdd2Impl;
    SubImpl = &Array<sp_t>::AvxSubImpl;
    Sub2Impl = &Array<sp_t>::AvxSub2Impl;
    MulImpl = &Array<sp_t>::AvxMulImpl;
    Mul2Impl = &Array<sp_t>::AvxMul2Impl;
    DivImpl = &Array<sp_t>::AvxDivImpl;
    Div2Impl = &Array<sp_t>::AvxDiv2Impl;
    SqrtImpl = &Array<sp_t>::AvxSqrtImpl;
  }

  template<>
  void Array<sp_t>::BuildAvx2ArchBinding()
  {
    AddImpl = &Array<sp_t>::Avx2AddImpl;
    Add2Impl = &Array<sp_t>::Avx2Add2Impl;
    SubImpl = &Array<sp_t>::Avx2SubImpl;
    Sub2Impl = &Array<sp_t>::Avx2Sub2Impl;
    MulImpl = &Array<sp_t>::Avx2MulImpl;
    Mul2Impl = &Array<sp_t>::Avx2Mul2Impl;
    DivImpl = &Array<sp_t>::Avx2DivImpl;
    Div2Impl = &Array<sp_t>::Avx2Div2Impl;
    SqrtImpl = &Array<sp_t>::Avx2SqrtImpl;
  }

  template<>
  void Array<sp_t>::BuildArchBinding()
  {
    if ( _procCaps.IsAvx() ) {
      BuildAvxArchBinding();
    } else if ( _procCaps.IsAvx2() ) {
      BuildAvx2ArchBinding();
    } else {
      BuildFallbackArchBinding();
    }
  }
}
