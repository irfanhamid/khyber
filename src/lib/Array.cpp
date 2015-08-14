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
  Array<sp_t> Array<sp_t>::ScalarMul(sp_t multiplier)
  {
    return (this->*ScalarMulImpl)(multiplier);
  }

  template<>
  Array<sp_t>& Array<sp_t>::TransformScalarMul(sp_t multiplier)
  {
    return (this->*TransformScalarMulImpl)(multiplier);
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
  Array<sp_t> Array<sp_t>::ScalarDiv(sp_t divisor)
  {
    return (this->*ScalarDivImpl)(divisor);
  }

  template<>
  Array<sp_t>& Array<sp_t>::TransformScalarDiv(sp_t divisor)
  {
    return (this->*TransformScalarDivImpl)(divisor);
  }

  template<>
  Array<sp_t> Array<sp_t>::Sqrt()
  {
    return (this->*SqrtImpl)();
  }

  template<>
  Array<sp_t>& Array<sp_t>::Sqrt(Array<sp_t>& src)
  {
    return (this->*Sqrt2Impl)(src);
  }

  template<>
  Array<sp_t> Array<sp_t>::Square()
  {
    return (this->*SquareImpl)();
  }

  template<>
  Array<sp_t>& Array<sp_t>::Square(Array<sp_t>& src)
  {
    return (this->*Square2Impl)(src);
  }

  template<>
  Array<sp_t> Array<sp_t>::Cube()
  {
    return (this->*CubeImpl)();
  }

  template<>
  Array<sp_t>& Array<sp_t>::Cube(Array<sp_t>& src)
  {
    return (this->*Cube2Impl)(src);
  }

  template<>
  sp_t Array<sp_t>::DotProduct(const Array<sp_t>& multiplicand) const
  {
    return (this->*DotProductImpl)(multiplicand);
  }

  template<>
  sp_t Array<sp_t>::Summation() const
  {
    return (this->*SummationImpl)();
  }

  template<>
  sp_t Array<sp_t>::Distance(const Array<sp_t>& v2) const
  {
    return (this->*DistanceImpl)(v2);
  }

  template<>
  Array<sp_t> Array<sp_t>::Reciprocate()
  {
    return (this->*ReciprocateImpl)();
  }

  template<>
  Array<sp_t>& Array<sp_t>::TransformReciprocate()
  {
    return (this->*TransformReciprocateImpl)();
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
  Array<sp_t>& Array<sp_t>::AvxTransformScalarMulImpl(sp_t multiplier)
  {
    avx::InternalScalarMul(this->size(),
                           multiplier,
                           this->data(),
                           this->data());
    return *this;
  }

  template<>
  Array<sp_t> Array<sp_t>::AvxScalarMulImpl(sp_t multiplier)
  {
    Array<sp_t> product(this->size());
    return std::move(AvxTransformScalarMulImpl(multiplier));
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
  Array<sp_t>& Array<sp_t>::AvxTransformScalarDivImpl(sp_t divisor)
  {
    avx::InternalScalarDiv(this->size(),
                           divisor,
                           this->data(),
                           this->data());
    return *this;
  }

  template<>
  Array<sp_t> Array<sp_t>::AvxScalarDivImpl(sp_t divisor)
  {
    Array<sp_t> quotient(this->size());
    return std::move(AvxTransformScalarDivImpl(divisor));
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

  template<>
  Array<sp_t>& Array<sp_t>::AvxSqrt2Impl(Array<sp_t>& src)
  {
    avx::InternalSqrt(this->size(),
                      this->data(),
                      src.data());
    return *this;
  }

  template<>
  Array<sp_t> Array<sp_t>::AvxSquareImpl()
  {
    Array<sp_t> result(this->size());
    avx::InternalSquare(this->size(),
                        result.data(),
                        this->data());
    return std::move(result);
  }

  template<>
  Array<sp_t>& Array<sp_t>::AvxSquare2Impl(Array<sp_t>& src)
  {
    avx::InternalSquare(this->size(),
                        this->data(),
                        this->data());
    return *this;
  }

  template<>
  Array<sp_t> Array<sp_t>::AvxCubeImpl()
  {
    Array<sp_t> result(this->size());
    avx::InternalCube(this->size(),
                      result.data(),
                      this->data());
    return std::move(result);
  }

  template<>
  Array<sp_t>& Array<sp_t>::AvxCube2Impl(Array<sp_t>& src)
  {
    avx::InternalCube(this->size(),
                      this->data(),
                      this->data());
    return *this;
  }

  template<>
  sp_t Array<sp_t>::AvxDotProductImpl(const Array<sp_t> &multiplicand) const
  {
    sp_t dotProduct = 0;
    avx::InternalDotProduct(this->size(),
                            &dotProduct,
                            this->data(),
                            multiplicand.data());
    return dotProduct;
  }

  template<>
  sp_t Array<sp_t>::AvxSummationImpl() const
  {
    sp_t sigma = 0;
    avx::InternalSummation(this->size(),
                           &sigma,
                           this->data());
    return sigma;
  }

  template<>
  Array<sp_t> Array<sp_t>::AvxNegateImpl()
  {
    Array<sp_t> negated(this->size());
    avx::InternalNegate(this->size(),
                        negated.data(),
                        this->data());
    return std::move(negated);
  }

  template<>
  Array<sp_t>& Array<sp_t>::AvxNegate2Impl(Array<sp_t>& src)
  {
    avx::InternalNegate(this->size(),
                        this->data(),
                        src.data());
    return *this;
  }

  template<>
  sp_t Array<sp_t>::AvxDistanceImpl(const Array<sp_t>& v2) const
  {
    sp_t distance = 0.0;
    avx::InternalDistance(this->size(),
                          &distance,
                          this->data(),
                          v2.data());
    return distance;
  }

  template<>
  Array<sp_t> Array<sp_t>::AvxReciprocateImpl()
  {
    Array<sp_t> reciprocal(this->size());
    return std::move(reciprocal.TransformReciprocate());
  }

  template<>
  Array<sp_t>& Array<sp_t>::AvxTransformReciprocateImpl()
  {
    avx::InternalReciprocate(this->size(),
                             this->data(),
                             this->data());
    return *this;
  }

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
  Array<ui32_t>& Array<ui32_t>::AvxAdd2Impl(Array<ui32_t> &augend,
                                            const Array<ui32_t> &addend)
  {
    avx::InternalAdd(this->_buffer.size(),
                     this->_buffer.data(),
                     augend._buffer.data(),
                     addend._buffer.data());
    return *this;
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
  Array<sp_t>& Array<sp_t>::Avx2Add2Impl(Array<sp_t> &augend,
					 const Array<sp_t> &addend)
  {
    avx2::InternalAdd(this->_buffer.size(),
		      this->_buffer.data(),
		      augend._buffer.data(),
		      addend._buffer.data());
    return *this;
  }
  
  template<>
  Array<sp_t> Array<sp_t>::Avx2NegateImpl()
  {
    Array<sp_t> negated(this->size());
    avx2::InternalNegate(this->size(),
                         negated.data(),
                         this->data());
    return std::move(negated);
  }

  template<>
  Array<sp_t>& Array<sp_t>::Avx2Negate2Impl(Array<sp_t>& src)
  {
    avx2::InternalNegate(this->size(),
                         this->data(),
                         src.data());
    return *this;
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
    Sqrt2Impl = &Array<sp_t>::FallbackSqrt2Impl;
    SquareImpl = &Array<sp_t>::FallbackSquareImpl;
    Square2Impl = &Array<sp_t>::FallbackSquare2Impl;
    CubeImpl = &Array<sp_t>::FallbackCubeImpl;
    Cube2Impl = &Array<sp_t>::FallbackCube2Impl;
    DotProductImpl = &Array<sp_t>::FallbackDotProductImpl;
    SummationImpl = &Array<sp_t>::FallbackSummationImpl;
    NegateImpl = &Array<sp_t>::FallbackNegateImpl;
    Negate2Impl = &Array<sp_t>::FallbackNegate2Impl;
    DistanceImpl = &Array<sp_t>::FallbackDistanceImpl;
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
    Sqrt2Impl = &Array<sp_t>::AvxSqrt2Impl;
    SquareImpl = &Array<sp_t>::AvxSquareImpl;
    Square2Impl = &Array<sp_t>::AvxSquare2Impl;
    CubeImpl = &Array<sp_t>::AvxCubeImpl;
    Cube2Impl = &Array<sp_t>::AvxCube2Impl;
    DotProductImpl = &Array<sp_t>::AvxDotProductImpl;
    SummationImpl = &Array<sp_t>::AvxSummationImpl;
    NegateImpl = &Array<sp_t>::AvxNegateImpl;
    Negate2Impl = &Array<sp_t>::AvxNegate2Impl;
    DistanceImpl = &Array<sp_t>::AvxDistanceImpl;
  }

  template<>
  void Array<sp_t>::BuildAvx2ArchBinding()
  {
    AddImpl = &Array<sp_t>::Avx2AddImpl;
    Add2Impl = &Array<sp_t>::Avx2Add2Impl;
    NegateImpl = &Array<sp_t>::Avx2NegateImpl;
    Negate2Impl = &Array<sp_t>::Avx2Negate2Impl;
  }

  template<>
  void Array<sp_t>::BuildArchBinding()
  {
    BuildFallbackArchBinding();

    if ( _procCaps.IsAvx() ) {
      BuildAvxArchBinding();
    } else if ( _procCaps.IsAvx2() ) {
      BuildAvx2ArchBinding();
    }
  }
}
