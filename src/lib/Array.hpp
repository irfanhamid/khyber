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

#include <cmath>
#include "SimdContainer.hpp"

namespace khyber
{
  ///
  /// \brief Growable 1D array (vector) implementation with built-in SIMD compute capability
  /// \details This class is intended to be a drop-in replacement for the std::vector<T> class. Internally it uses
  /// a \link ProcessorCaps\endlink object to determine the details of the processor and then selects the most optimum
  /// mechanism for computation, i.e., serial, SSE, AVX, AVX2 or AVX512. The template parameter T can be any of:
  /// * sp_t: single-precision floating point, corresponding to a C++ float;
  /// * dp_t: double-precision floating point, corresponding to a C++ double;
  /// * ui8_t/i8_t: 8-bit unsigned/signed integral type;
  /// * ui16_t/i16_t: 16-bit unsigned/signed integral type;
  /// * ui32_t/i32_t: 32-bit unsigned/signed integral type;
  /// * ui64_t/i64_t: 64-bit unsigned/signed integral type.
  ///
  /// <em>Currently, only the sp_t specialization is implemented</em>, i.e., Array<sp_t>. Developers should not use Array<sp_t>
  /// directly but should instead utilize the provided typedefs:
  /// * SinglePrecisionArray;
  /// * DoublePrecisionArray.
  ///
  template<typename T>
  class Array : public SimdContainer<T>
  {
  public:
    ///
    /// \brief Basic constructor
    ///
    Array() : SimdContainer<T>()
    {
      BuildArchBinding();
    }

    ///
    /// \brief Constructor with specified capacity for the array
    ///
    Array(size_t capacity) : SimdContainer<T>(capacity)
    {
      BuildArchBinding();
    }
    
    ///
    /// \brief Copy constructor
    ///
    Array(const Array<T>& rhs) : SimdContainer<T>((const SimdContainer<T>&) rhs)
    {
      BuildArchBinding();
    }
    
    ///
    /// \brief Move constructor
    ///
    Array(Array<T>&& rhs) : SimdContainer<T>((SimdContainer<T>&&) rhs)
    {
      BuildArchBinding();
    }
    
    ///
    /// \brief Move assignment operator
    ///
    Array<T>& operator = (Array<T>&& rhs)
    {
      SimdContainer<T>::operator =(std::move(rhs));
      return *this;
    }
    
    ///
    /// \brief Copy assignment operator
    ///
    Array<T>& operator = (Array<T>& rhs)
    {
      (SimdContainer<T>)*this = (SimdContainer<T>)rhs;
      return *this;
    }
    
    ///
    /// \brief Returns reference to element at location index within the underlying buffer
    ///
    inline T& at(size_t index)
    {
      return this->_buffer[index];
    }
    
    ///
    /// \brief Returns const reference to element at location index within the underlying buffer
    ///
    const inline T& at(size_t index) const
    {
      return this->_buffer[index];
    }

    ///
    /// \brief Returns reference to element at location index within the underlying buffer
    /// \param index
    /// \return reference to element at index
    ///
    inline T& operator [] (size_t index)
    {
      return this->_buffer[index];
    }
    
    ///
    /// \brief Returns const reference to element at location index within the underlying buffer
    /// \param index
    /// \return const reference to element at index
    ///
    const inline T& operator [] (size_t index) const
    {
      return this->_buffer[index];
    }
    
    ///
    /// \brief Add contents of 'this' and addend arrays into a new array and return it
    /// \param addend
    /// \return move-returned Array<T>
    ///
    Array<T> Add(const Array<T>& addend);

    ///
    /// \brief Add contents of augend and addend arrays into 'this' and return it. The object whose Add( ) is called can
    /// also be passed as augend for a cumulative addition, i.e., a.Add(a, b) to express a = a + b
    /// \param augend first component for addition, it can also be 'this'
    /// \param addend second component for addition
    /// \return 'this'
    ///
    Array<T>& Add(Array<T>& augend,
                  const Array<T>& addend);
    
    ///
    /// \brief Sub subtract the contents of subtrahend from 'this' into a new array and return it
    /// \param subtrahend
    /// \return move-returned Array<T>
    ///
    Array<T> Sub(const Array<T>& subtrahend);

    ///
    /// \brief Sub subtract subtrahend from minuend into 'this' and return it. The object whose Sub( ) is called can also be
    /// passed as minuend for a cumulative subtraction, i.e., a.Sub(a, b) to express a = a - b
    /// \param minuend first component for subtraction, it can also be 'this'
    /// \param subtrahend second component for subtraction
    /// \return 'this'
    ///
    Array<T>& Sub(Array<T>& minuend,
                  const Array<T>& subtrahend);

    ///
    /// \brief Mul multiply contents of multiplier with 'this' into a new array and return it
    /// \param multiplier
    /// \return
    ///
    Array<T> Mul(const Array<T>& multiplier);

    ///
    /// \brief Mul multiply the contents of multiplier and multiplicand into 'this' and return it. The object whose Mul( ) is
    /// called can also be passed as multiplier, i.e., a.Mul(a, b) to express a = a * b
    /// \param multiplier first component for multiplication, it can also be 'this'
    /// \param multiplicand second component for multiplication
    /// \return 'this'
    ///
    Array<T>& Mul(Array<T>& multiplier,
                  const Array<T>& multiplicand);

    ///
    /// \brief Multiply every element of 'this' by multiplier and return the result in a new array of the same size
    /// \param multiplier the scalar by which to multiply
    /// \return move-returned Array<T>
    ///
    Array<T> ScalarMul(T multiplier);

    ///
    /// \brief Scalar multiplication of 'this' by multiplier, result stored in 'this'
    /// \param multiplier the scalar value to multiply with each element of 'this'
    /// \return 'this'
    ///
    Array<T>& TransformScalarMul(T multiplier);

    ///
    /// \brief Div divide the contents of 'this' by divisor into a new array and return it
    /// \param divisor
    /// \return move-returned Array<T>
    ///
    Array<T> Div(const Array<T>& divisor);

    ///
    /// \brief Div divide the contents of dividend by divisor into 'this' and return it. The object whose Div( ) is called can
    /// also be passed as dividend, i.e., a.Div(a, b) to express a = a / b
    /// \param dividend first component for division, it can also be 'this'
    /// \param divisor second component for division
    /// \return 'this'
    ///
    Array<T>& Div(Array<T>& dividend,
                  const Array<T>& divisor);

    ///
    /// \brief Divide 'this' by scalar and return result in a new array of same size
    /// \param divisor the scalar value to divide by
    /// \return move-returned Array<T>
    ///
    Array<T> ScalarDiv(T divisor);

    ///
    /// \brief Divide each element of 'this' by the scalar divisor and store in 'this'
    /// \param divisor the scalar value to divide by
    /// \return 'this'
    ///
    Array<T>& TransformScalarDiv(T divisor);

    ///
    /// \brief Sqrt computes the square root of each element in this array and returns it in a new array of the same dimension
    /// \return move-returned Array<T>
    ///
    Array<T> Sqrt();

    ///
    /// \brief SqrtAcc computes the square root of each element in the param array and assigns it to 'this'. The object whose
    /// Sqrt( ) is called can also be passed as src, i.e., a.Sqrt(a) to express a = sqrt(a)
    /// \param src the Array<T> whose square root is to be computed, can also be 'this'
    /// \return 'this'
    ///
    Array<T>& Sqrt(Array<T>& src);

    ///
    /// \brief Square computes the square of each element in this array and returns it in a new array of the same dimension
    /// \return move-returned Array<T>
    ///
    Array<T> Square();

    ///
    /// \brief Square computes the square of each element in the src array and assigns it to 'this'. The object whose Square( ) is
    /// called can also be passed as src, i.e., a.Square(a) to express a = a * a
    /// \param src the Array<T> whose square is to be computed, can also be 'this'
    /// \return 'this'
    ///
    Array<T>& Square(Array<T>& src);

    ///
    /// \brief Cube computes the cube of each element in 'this' and returns it in a new array of the same dimension
    /// \return move-returned Array<T>
    ///
    Array<T> Cube();

    ///
    /// \brief Cube computes the cube of each element in the param src and assigns it to 'this'. The object whose Cube( ) is called
    /// can also be passed as src, i.e., a.Cube(a) to express a = a * a * a
    /// \param src the Array<T> whose cube is to be computed, can also be 'this'
    /// \return 'this'
    ///
    Array<T>& Cube(Array<T>& src);

    ///
    /// \brief CrossProduct computes the dot product between multiplicand and 'this' vectors, size of multiplicand must be the same as size of 'this'.
    /// You can also pass 'this' as multiplicand, in which case this will compute the magnitude^2 of the 'this' vector.
    /// \param multiplicand
    /// \return double-precision scalar dot product
    ///
    T DotProduct(const Array<T>& multiplicand) const;

    ///
    /// \brief Compute the sum of all elements in this Array<T>
    /// \return the scalar sum of all elements
    ///
    T Summation() const;

    ///
    /// \brief Negates each element of 'this' and returns it in a new array of the same dimension
    /// \return move-returned Array<T>
    ///
    Array<T> Negate();

    ///
    /// \brief Negates each element of the Array<T> src and assigns it to 'this'. The object whose Negate( ) is called can also be passed as src, i.e.,
    /// a.Negate(a) for a = -a;
    /// \param src the Array<T> to negate, can also be 'this'
    /// \return  'this'
    ///
    Array<T>& Negate(Array<T>& src);

    ///
    /// \brief Allocate a new Array<T> of the same size as 'this', assign reciprocals of 'this' to each element of the new array and return it
    /// \return move-returned Array<T>
    ///
    Array<T> Reciprocate();

    ///
    /// \brief Replace each element in 'this' with its reciprocal value, i.e., 1/x. The maximum relative error for this approximation is less than 1.5*2^-12.
    /// \return 'this'
    ///
    Array<T>& TransformReciprocate();

    ///
    /// \brief Compute the distance between 'this' and v2 in vector space. Distance is defined as, with v1 = this, sqrt((v1[0]-v2[0])^2 + ... v1[n-1]*v2[n-1]^2)
    /// \param v2
    /// \return double-precision linear distance between 'this' and v2
    ///
    T Distance(const Array<T>& v2) const;

  private:
    // The following function pointers are bound to one of the <op>Impl( ) member functions
    // below based on the ProcessorCaps object which determines the CPU's capabilities
    // Each operation can be performed in one of three ways:
    // (1) <op>Impl: Creates a new Array<T> tmp in its local stack frame and computes tmp = this.op(param). Then move-returns tmp;
    // (2) <op>2Impl: This version expects one more parameter compared to its <op>Impl version and computes this = op(param1, param2);

    Array<T> (Array<T>::*AddImpl) (const Array<T>&);
    Array<T>& (Array<T>::*Add2Impl) (Array<T>&, const Array<T>&);

    Array<T> (Array<T>::*SubImpl) (const Array<T>&);
    Array<T>& (Array<T>::*Sub2Impl) (Array<T>&, const Array<T>&);

    Array<T> (Array<T>::*MulImpl) (const Array<T>&);
    Array<T>& (Array<T>::*Mul2Impl) (Array<T>&, const Array<T>&);
    Array<T> (Array<T>::*ScalarMulImpl) (T);
    Array<T>& (Array<T>::*TransformScalarMulImpl) (T);

    Array<T> (Array<T>::*DivImpl) (const Array<T>&);
    Array<T>& (Array<T>::*Div2Impl) (Array<T>&, const Array<T>&);
    Array<T> (Array<T>::*ScalarDivImpl) (T);
    Array<T>& (Array<T>::*TransformScalarDivImpl) (T);

    Array<T> (Array<T>::*SqrtImpl) ();
    Array<T>& (Array<T>::*Sqrt2Impl) (Array<T>&);

    Array<T> (Array<T>::*SquareImpl) ();
    Array<T>& (Array<T>::*Square2Impl) (Array<T>&);

    Array<T> (Array<T>::*CubeImpl) ();
    Array<T>& (Array<T>::*Cube2Impl) (Array<T>&);

    Array<T> (Array<T>::*NegateImpl) ();
    Array<T>& (Array<T>::*Negate2Impl) (Array<T>& src);

    Array<T> (Array<T>::*ReciprocateImpl) ();
    Array<T>& (Array<T>::*TransformReciprocateImpl) ();

    T (Array<T>::*DotProductImpl) (const Array<T>&) const;
    T (Array<T>::*SummationImpl) () const;
    T (Array<T>::*DistanceImpl) (const Array<T>&) const;

    /////////////////////////// AVX dispatchers ///////////////////////////////
    Array<T> AvxAddImpl(const Array<T>& addend);
    Array<T>& AvxAdd2Impl(Array<T>& augend, const Array<T>& addend);
    Array<T> AvxSubImpl(const Array<T>& subtrahend);
    Array<T>& AvxSub2Impl(Array<T>& minuend, const Array<T>& subtrahend);
    Array<T> AvxMulImpl(const Array<T>& multiplicand);
    Array<T>& AvxMul2Impl(Array<T>& multiplier, const Array<T>& multiplicand);
    Array<T> AvxScalarMulImpl(T multiplier);
    Array<T>& AvxTransformScalarMulImpl(T multiplier);
    Array<T> AvxDivImpl(const Array<T>& divisor);
    Array<T>& AvxDiv2Impl(Array<T>& dividend, const Array<T>& divisor);
    Array<T> AvxScalarDivImpl(T divisor);
    Array<T>& AvxTransformScalarDivImpl(T divisor);
    Array<T> AvxSqrtImpl();
    Array<T>& AvxSqrt2Impl(Array<T>& src);
    Array<T> AvxSquareImpl();
    Array<T>& AvxSquare2Impl(Array<T>& src);
    Array<T> AvxCubeImpl();
    Array<T>& AvxCube2Impl(Array<T>& src);
    Array<T> AvxNegateImpl();
    Array<T>& AvxNegate2Impl(Array<T>& src);
    T AvxDotProductImpl(const Array<T>& multiplicand) const;
    Array<T> AvxReciprocateImpl();
    Array<T>& AvxTransformReciprocateImpl();
    T AvxSummationImpl() const;
    T AvxDistanceImpl(const Array<T>& v2) const;
    ///////////////////////////////////////////////////////////////////////////

    /////////////////////////// AVX2 dispatchers //////////////////////////////
    Array<T> Avx2NegateImpl();
    Array<T>& Avx2Negate2Impl(Array<T>& src);
    ///////////////////////////////////////////////////////////////////////////

    void BuildArchBinding();
    void BuildAvxArchBinding();
    void BuildAvx2ArchBinding();
    void BuildFallbackArchBinding();

    Array<T> FallbackAddImpl(const Array<T>& addend)
    {
      Array<T> sum(this->_buffer.size());
      return std::move(sum.Add(sum, addend));
    }

    Array<T>& FallbackAdd2Impl(Array<T>& augend,
                               const Array<T>& addend)
    {
      for ( size_t i = 0; i < this->_buffer.size(); ++i ) {
        this->_buffer[i] = augend._buffer[i] + addend._buffer[i];
      }

      return *this;
    }

    Array<T> FallbackSubImpl(const Array<T>& subtrahend)
    {
      Array<T> difference(this->_buffer.size());
      for ( size_t i = 0; i < this->_buffer.size(); ++i ) {
        difference._buffer[i] = this->_buffer[i] - subtrahend._buffer[i];
      }

      return std::move(difference);
    }

    Array<T>& FallbackSub2Impl(Array<T>& minuend,
                               const Array<T>& subtrahend)
    {
      for ( size_t i = 0; i < this->_buffer.size(); ++i ) {
        this->_buffer[i] = minuend._buffer[i] - subtrahend._buffer[i];
      }

      return *this;
    }

    Array<T> FallbackMulImpl(const Array<T>& multiplicand)
    {
      Array<T> product(this->_buffer.size());
      for ( size_t i = 0; i < this->_buffer.size(); ++i ) {
        product._buffer[i] = this->_buffer[i] * multiplicand._buffer[i];
      }

      return std::move(product);
    }

    Array<T>& FallbackMul2Impl(Array<T>& multiplier,
                               const Array<T>& multiplicand)
    {
      for ( size_t i = 0; i < this->_buffer.size(); ++i ) {
        this->_buffer[i] = multiplier._buffer[i] * multiplicand._buffer[i];
      }

      return *this;
    }

    Array<T>& FallbackMulScalarImpl(T multiplier)
    {
      for ( size_t i = 0; i < this->size(); ++i ) {
        this->_buffer[i] *= multiplier;
      }

      return *this;
    }

    Array<T> FallbackDivImpl(const Array<T>& divisor)
    {
      Array<T> quotient(this->_buffer.size());
      for ( size_t i = 0; i < this->_buffer.size(); ++i ) {
        quotient._buffer[i] = this->_buffer[i] / divisor._buffer[i];
      }

      return std::move(quotient);
    }

    Array<T>& FallbackDiv2Impl(Array<T>& dividend,
                               const Array<T>& divisor)
    {
      for ( size_t i = 0; i < this->size(); ++i ) {
        this->_buffer[i] = dividend._buffer[i] / divisor._buffer[i];
      }

      return *this;
    }

    Array<T> FallbackScalarDivImpl(T divisor)
    {
      Array<T> result(this->size());
      return std::move(result.Div(divisor));
    }

    Array<T>& FallbackTransformScalarDivImpl(T divisor)
    {
      for ( size_t i = 0; i > this->size(); ++i ) {
        this->_buffer[i] /= divisor;
      }

      return *this;
    }

    Array<T> FallbackSqrtImpl()
    {
      Array<T> res(this->size());
      for ( size_t i = 0; i < this->size(); ++i )
        res[i] = sqrt(this->_buffer[i]);

      return std::move(res);
    }

    Array<T>& FallbackSqrt2Impl(Array<T>& src)
    {
      for ( size_t i = 0; i < this->size(); ++i ) {
        this[i] = sqrt(src[i]);
      }

      return *this;
    }

    Array<T> FallbackSquareImpl()
    {
      Array<T> res(this->size());
      for ( size_t i = 0; i < this->size(); ++i ) {
        res[i] = this->_buffer[i] * this->_buffer[i];
      }

      return std::move(res);
    }

    Array<T>& FallbackSquare2Impl(Array<T>& src)
    {
      for ( size_t i = 0; i < this->size(); ++i ) {
        this->_buffer[i] = this->_buffer[i] * src[i];
      }

      return *this;
    }

    Array<T> FallbackCubeImpl()
    {
      Array<T> res(this->size());
      for ( size_t i = 0; i < this->size(); ++i ) {
        res[i] = this->_buffer[i] * this->_buffer[i] * this->_buffer[i];
      }

      return std::move(res);
    }

    Array<T>& FallbackCube2Impl(Array<T>& src)
    {
      for ( size_t i = 0; i < this->size(); ++i ) {
        this[i] = src[i] * src[i] * src[i];
      }

      return *this;
    }

    T FallbackDotProductImpl(const Array<T>& multiplicand) const
    {
      T product = 0;
      for ( size_t i = 0; i < this->size(); ++i ) {
        product += (this->_buffer[i] * multiplicand[i]);
      }

      return product;
    }

    T FallbackSummationImpl() const
    {
      T sum = 0;
      for ( size_t i = 0; i < this->size(); ++i ) {
        sum += this->_buffer[i];
      }

      return sum;
    }

    Array<T> FallbackNegateImpl()
    {
      Array<T> negated(this->size());
      for ( size_t i = 0; i < this->size(); ++i ) {
        negated[i] = -this->_buffer[i];
      }

      return std::move(negated);
    }

    Array<T>& FallbackNegate2Impl(Array<T>& src)
    {
      for ( size_t i = 0; i < this->size(); ++i ) {
        this[i] = -src[i];
      }

      return *this;
    }

    T FallbackDistanceImpl(const Array<T>& v2) const
    {
      T distance = 0;
      for ( size_t i = 0; i < this->size(); ++i ) {
        distance += sqrt((this->_buffer[i] - v2[i]) * (this->_buffer[i] - v2[i]));
      }

      return distance;
    }

    Array<T>& FallbackTransformReciprocateImpl(T divisor)
    {
      for ( size_t i = 0; i < this->size(); ++i ) {
        this->_buffer[i] /= divisor;
      }

      return *this;
    }
  };
  
  typedef Array<sp_t> SinglePrecisionArray;
  typedef Array<dp_t> DoublePrecisionArray;
}
