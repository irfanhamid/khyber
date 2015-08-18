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
  Array<float> Array<float>::Add(const Array<float>& addend)
  {
    // This ugliness (and the one in AddAcc( ) is for pointer-to-member-function
    // I know we could use boost::function, and I did, but that is much slower
    return (this->*AddImpl)(addend);
  }
  
  template<>
  Array<float>& Array<float>::Add(Array<float> &augend,
				  const Array<float> &addend)
  {
    return (this->*Add2Impl)(augend, addend);
  }

  template<>
  Array<float> Array<float>::Sub(const Array<float>& subtrahend)
  {
    return (this->*SubImpl)(subtrahend);
  }

  template<>
  Array<float>& Array<float>::Sub(Array<float> &minuend,
				  const Array<float> &subtrahend)
  {
    return (this->*Sub2Impl)(minuend, subtrahend);
  }

  template<>
  Array<float> Array<float>::Mul(const Array<float> &multiplier)
  {
    return (this->*MulImpl)(multiplier);
  }

  template<>
  Array<float>& Array<float>::Mul(Array<float> &multiplier,
				  const Array<float> &multiplicand)
  {
    return (this->*Mul2Impl)(multiplier, multiplicand);
  }

  template<>
  Array<float> Array<float>::ScalarMul(float multiplier)
  {
    return (this->*ScalarMulImpl)(multiplier);
  }

  template<>
  Array<float>& Array<float>::TransformScalarMul(float multiplier)
  {
    return (this->*TransformScalarMulImpl)(multiplier);
  }

  template<>
  Array<float> Array<float>::Div(const Array<float> &divisor)
  {
    return (this->*DivImpl)(divisor);
  }

  template<>
  Array<float>& Array<float>::Div(Array<float> &dividend,
				  const Array<float> &divisor)
  {
    return (this->*Div2Impl)(dividend, divisor);
  }

  template<>
  Array<float> Array<float>::ScalarDiv(float divisor)
  {
    return (this->*ScalarDivImpl)(divisor);
  }

  template<>
  Array<float>& Array<float>::TransformScalarDiv(float divisor)
  {
    return (this->*TransformScalarDivImpl)(divisor);
  }

  template<>
  Array<float> Array<float>::Sqrt()
  {
    return (this->*SqrtImpl)();
  }

  template<>
  Array<float>& Array<float>::Sqrt(Array<float>& src)
  {
    return (this->*Sqrt2Impl)(src);
  }

  template<>
  Array<float> Array<float>::Square()
  {
    return (this->*SquareImpl)();
  }

  template<>
  Array<float>& Array<float>::Square(Array<float>& src)
  {
    return (this->*Square2Impl)(src);
  }

  template<>
  Array<float> Array<float>::Cube()
  {
    return (this->*CubeImpl)();
  }

  template<>
  Array<float>& Array<float>::Cube(Array<float>& src)
  {
    return (this->*Cube2Impl)(src);
  }

  template<>
  float Array<float>::DotProduct(const Array<float>& multiplicand) const
  {
    return (this->*DotProductImpl)(multiplicand);
  }

  template<>
  float Array<float>::Summation() const
  {
    return (this->*SummationImpl)();
  }

  template<>
  float Array<float>::Distance(const Array<float>& v2) const
  {
    return (this->*DistanceImpl)(v2);
  }

  template<>
  Array<float> Array<float>::Reciprocate()
  {
    return (this->*ReciprocateImpl)();
  }

  template<>
  Array<float>& Array<float>::TransformReciprocate()
  {
    return (this->*TransformReciprocateImpl)();
  }

  /////////////////////////////////////////////////////////////////////////////
  


  /////////////////////// AVX implementation dispatchers //////////////////////

  template<>
  Array<float> Array<float>::AvxAddImpl(const Array<float>& addend)
  {
    Array<float> sum(this->_buffer.size());
    avx::InternalAdd(this->_buffer.size(),
                     sum._buffer.data(),
                     this->_buffer.data(),
                     addend._buffer.data());
    return std::move(sum);
  }

  template<>
  Array<float>& Array<float>::AvxAdd2Impl(Array<float> &augend,
					  const Array<float> &addend)
  {
    avx::InternalAdd(this->_buffer.size(),
                     this->_buffer.data(),
                     augend._buffer.data(),
                     addend._buffer.data());
    return *this;
  }
  
  template<>
  Array<float> Array<float>::AvxSubImpl(const Array<float> &subtrahend)
  {
    Array<float> difference(this->_buffer.size());
    avx::InternalSub(this->_buffer.size(),
                     difference._buffer.data(),
                     this->_buffer.data(),
                     subtrahend._buffer.data());
    return std::move(difference);
  }

  template<>
  Array<float>& Array<float>::AvxSub2Impl(Array<float>& minuend,
					  const Array<float>& subtrahend)
  {
    avx::InternalSub(this->size(),
                     this->data(),
                     minuend.data(),
                     subtrahend.data());
    return *this;
  }

  template<>
  Array<float> Array<float>::AvxMulImpl(const Array<float> &multiplier)
  {
    Array<float> product(this->size());
    avx::InternalMul(this->size(),
                     product.data(),
                     this->data(),
                     multiplier.data());
    return std::move(product);
  }

  template<>
  Array<float>& Array<float>::AvxMul2Impl(Array<float> &multiplier,
					  const Array<float> &multiplicand)
  {
    avx::InternalMul(this->size(),
                     this->data(),
                     multiplier.data(),
                     multiplicand.data());
    return *this;
  }

  template<>
  Array<float>& Array<float>::AvxTransformScalarMulImpl(float multiplier)
  {
    avx::InternalScalarMul(this->size(),
                           multiplier,
                           this->data(),
                           this->data());
    return *this;
  }

  template<>
  Array<float> Array<float>::AvxScalarMulImpl(float multiplier)
  {
    Array<float> product(this->size());
    return std::move(AvxTransformScalarMulImpl(multiplier));
  }

  template<>
  Array<float> Array<float>::AvxDivImpl(const Array<float> &divisor)
  {
    Array<float> quotient(divisor.size());
    avx::InternalDiv(this->size(),
                     quotient.data(),
                     this->data(),
                     divisor.data());
    return std::move(quotient);
  }

  template<>
  Array<float>& Array<float>::AvxDiv2Impl(Array<float> &dividend,
					  const Array<float> &divisor)
  {
    avx::InternalDiv(this->size(),
                     this->data(),
                     dividend.data(),
                     divisor.data());
    return *this;
  }

  template<>
  Array<float>& Array<float>::AvxTransformScalarDivImpl(float divisor)
  {
    avx::InternalScalarDiv(this->size(),
                           divisor,
                           this->data(),
                           this->data());
    return *this;
  }

  template<>
  Array<float> Array<float>::AvxScalarDivImpl(float divisor)
  {
    Array<float> quotient(this->size());
    return std::move(AvxTransformScalarDivImpl(divisor));
  }

  template<>
  Array<float> Array<float>::AvxSqrtImpl()
  {
    Array<float> result(this->_buffer.size());
    avx::InternalSqrt(this->_buffer.size(),
                      result._buffer.data(),
                      this->_buffer.data());
    return std::move(result);
  }

  template<>
  Array<float>& Array<float>::AvxSqrt2Impl(Array<float>& src)
  {
    avx::InternalSqrt(this->size(),
                      this->data(),
                      src.data());
    return *this;
  }

  template<>
  Array<float> Array<float>::AvxSquareImpl()
  {
    Array<float> result(this->size());
    avx::InternalSquare(this->size(),
                        result.data(),
                        this->data());
    return std::move(result);
  }

  template<>
  Array<float>& Array<float>::AvxSquare2Impl(Array<float>& src)
  {
    avx::InternalSquare(this->size(),
                        this->data(),
                        this->data());
    return *this;
  }

  template<>
  Array<float> Array<float>::AvxCubeImpl()
  {
    Array<float> result(this->size());
    avx::InternalCube(this->size(),
                      result.data(),
                      this->data());
    return std::move(result);
  }

  template<>
  Array<float>& Array<float>::AvxCube2Impl(Array<float>& src)
  {
    avx::InternalCube(this->size(),
                      this->data(),
                      this->data());
    return *this;
  }

  template<>
  float Array<float>::AvxDotProductImpl(const Array<float> &multiplicand) const
  {
    float dotProduct = 0;
    avx::InternalDotProduct(this->size(),
                            &dotProduct,
                            this->data(),
                            multiplicand.data());
    return dotProduct;
  }

  template<>
  float Array<float>::AvxSummationImpl() const
  {
    float sigma = 0;
    avx::InternalSummation(this->size(),
                           &sigma,
                           this->data());
    return sigma;
  }

  template<>
  Array<float> Array<float>::AvxNegateImpl()
  {
    Array<float> negated(this->size());
    avx::InternalNegate(this->size(),
                        negated.data(),
                        this->data());
    return std::move(negated);
  }

  template<>
  Array<float>& Array<float>::AvxNegate2Impl(Array<float>& src)
  {
    avx::InternalNegate(this->size(),
                        this->data(),
                        src.data());
    return *this;
  }

  template<>
  float Array<float>::AvxDistanceImpl(const Array<float>& v2) const
  {
    float distance = 0.0;
    avx::InternalDistance(this->size(),
                          &distance,
                          this->data(),
                          v2.data());
    return distance;
  }

  template<>
  Array<float> Array<float>::AvxReciprocateImpl()
  {
    Array<float> reciprocal(this->size());
    return std::move(reciprocal.TransformReciprocate());
  }

  template<>
  Array<float>& Array<float>::AvxTransformReciprocateImpl()
  {
    avx::InternalReciprocate(this->size(),
                             this->data(),
                             this->data());
    return *this;
  }

  /////////////////////////////////////////////////////////////////////////////



  ////////////////////// AVX2 implementation dispatchers //////////////////////

  template<>
  Array<float> Array<float>::Avx2AddImpl(const Array<float>& addend)
  {
    Array<float> sum(this->_buffer.size());
    avx2::InternalAdd(this->_buffer.size(),
		      sum._buffer.data(),
		      this->_buffer.data(),
		      addend._buffer.data());
    return std::move(sum);
  }

  template<>
  Array<float>& Array<float>::Avx2Add2Impl(Array<float> &augend,
					   const Array<float> &addend)
  {
    avx2::InternalAdd(this->_buffer.size(),
		      this->_buffer.data(),
		      augend._buffer.data(),
		      addend._buffer.data());
    return *this;
  }
  
  template<>
  Array<float> Array<float>::Avx2NegateImpl()
  {
    Array<float> negated(this->size());
    avx2::InternalNegate(this->size(),
                         negated.data(),
                         this->data());
    return std::move(negated);
  }

  template<>
  Array<float>& Array<float>::Avx2Negate2Impl(Array<float>& src)
  {
    avx2::InternalNegate(this->size(),
                         this->data(),
                         src.data());
    return *this;
  }

  /////////////////////////////////////////////////////////////////////////////

  template<>
  void Array<float>::BuildFallbackArchBinding()
  {
    AddImpl = &Array<float>::FallbackAddImpl;
    Add2Impl = &Array<float>::FallbackAdd2Impl;
    SubImpl = &Array<float>::FallbackSubImpl;
    Sub2Impl = &Array<float>::FallbackSub2Impl;
    MulImpl = &Array<float>::FallbackMulImpl;
    Mul2Impl = &Array<float>::FallbackMul2Impl;
    DivImpl = &Array<float>::FallbackDivImpl;
    Div2Impl = &Array<float>::FallbackDiv2Impl;
    SqrtImpl = &Array<float>::FallbackSqrtImpl;
    Sqrt2Impl = &Array<float>::FallbackSqrt2Impl;
    SquareImpl = &Array<float>::FallbackSquareImpl;
    Square2Impl = &Array<float>::FallbackSquare2Impl;
    CubeImpl = &Array<float>::FallbackCubeImpl;
    Cube2Impl = &Array<float>::FallbackCube2Impl;
    DotProductImpl = &Array<float>::FallbackDotProductImpl;
    SummationImpl = &Array<float>::FallbackSummationImpl;
    NegateImpl = &Array<float>::FallbackNegateImpl;
    Negate2Impl = &Array<float>::FallbackNegate2Impl;
    DistanceImpl = &Array<float>::FallbackDistanceImpl;
  }

  template<>
  void Array<float>::BuildAvxArchBinding()
  {
    AddImpl = &Array<float>::AvxAddImpl;
    Add2Impl = &Array<float>::AvxAdd2Impl;
    SubImpl = &Array<float>::AvxSubImpl;
    Sub2Impl = &Array<float>::AvxSub2Impl;
    MulImpl = &Array<float>::AvxMulImpl;
    Mul2Impl = &Array<float>::AvxMul2Impl;
    DivImpl = &Array<float>::AvxDivImpl;
    Div2Impl = &Array<float>::AvxDiv2Impl;
    SqrtImpl = &Array<float>::AvxSqrtImpl;
    Sqrt2Impl = &Array<float>::AvxSqrt2Impl;
    SquareImpl = &Array<float>::AvxSquareImpl;
    Square2Impl = &Array<float>::AvxSquare2Impl;
    CubeImpl = &Array<float>::AvxCubeImpl;
    Cube2Impl = &Array<float>::AvxCube2Impl;
    DotProductImpl = &Array<float>::AvxDotProductImpl;
    SummationImpl = &Array<float>::AvxSummationImpl;
    NegateImpl = &Array<float>::AvxNegateImpl;
    Negate2Impl = &Array<float>::AvxNegate2Impl;
    DistanceImpl = &Array<float>::AvxDistanceImpl;
  }

  template<>
  void Array<float>::BuildAvx2ArchBinding()
  {
    AddImpl = &Array<float>::Avx2AddImpl;
    Add2Impl = &Array<float>::Avx2Add2Impl;
    NegateImpl = &Array<float>::Avx2NegateImpl;
    Negate2Impl = &Array<float>::Avx2Negate2Impl;
  }

  template<>
  void Array<float>::BuildArchBinding()
  {
    BuildFallbackArchBinding();

    if ( _procCaps.IsAvx() ) {
      BuildAvxArchBinding();
    } else if ( _procCaps.IsAvx2() ) {
      BuildAvx2ArchBinding();
    }
  }
}
