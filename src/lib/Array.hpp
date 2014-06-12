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
  /// Provides a growable vector implementation with built-in SIMD compute capability
  ///
  template<typename T>
  class Array : public SimdContainer<T>
  {
  public:
    ///
    /// Basic constructor
    ///
    Array() : SimdContainer<T>()
    {
      BuildArchBinding();
    }

    ///
    /// Constructor with specified capacity for the array
    ///
    Array(size_t capacity) : SimdContainer<T>(capacity)
    {
      BuildArchBinding();
    }
    
    ///
    /// Copy constructor
    ///
    Array(const Array<T>& rhs) : SimdContainer<T>((const SimdContainer<T>&) rhs)
    {
      BuildArchBinding();
    }
    
    ///
    /// Move constructor
    ///
    Array(Array<T>&& rhs) : SimdContainer<T>((SimdContainer<T>&&) rhs)
    {
      BuildArchBinding();
    }
    
    ///
    /// Move assignment operator
    ///
    Array<T>& operator = (Array<T>&& rhs)
    {
      SimdContainer<T>::operator =(std::move(rhs));
      return *this;
    }
    
    ///
    /// Copy assignment operator
    ///
    Array<T>& operator = (Array<T>& rhs)
    {
      (SimdContainer<T>)*this = (SimdContainer<T>)rhs;
      return *this;
    }
    
    ///
    /// Return the dimension (size) of this array
    ///
    inline size_t Dimension() const
    {
      return this->Size();
    }
    
    ///
    /// Returns reference to element at location index within the underlying buffer
    ///
    inline T& At(size_t index)
    {
      return this->_buffer[index];
    }
    
    ///
    /// Returns const reference to element at location index within the underlying buffer
    ///
    const inline T& At(size_t index) const
    {
      return this->_buffer[index];
    }

    ///
    /// Returns reference to element at location index within the underlying buffer
    ///
    inline T& operator [] (size_t index)
    {
      return this->_buffer[index];
    }
    
    ///
    /// Returns const reference to element at location index within the underlying buffer
    ///
    const inline T& operator [] (size_t index) const
    {
      return this->_buffer[index];
    }
    
    ///
    /// Add contents of 'this' and addend arrays into a new array and return it
    /// Consider using AddAcc() instead.
    ///
    Array<T> Add(const Array<T>& addend) const;

    ///
    /// \brief Add add contents of augend and addend arrays into 'this' and return it
    /// \param augend
    /// \param addend
    /// \return
    ///
    Array<T>& Add(const Array<T>& augend,
                  const Array<T>& addend);
    
    ///
    /// \brief AddAcc Add contents of addend into 'this' and return it
    /// \param addend Array to add to 'this'
    /// \return 'this'
    ///
    Array<T>& AddAcc(const Array<T>& addend);

    ///
    /// \brief Sub subtract the contents of subtrahend from 'this' into a new array and return it
    /// \param subtrahend
    /// \return 'this'
    ///
    Array<T> Sub(const Array<T>& subtrahend) const;

    ///
    /// \brief Sub subtract subtrahend from minuend into 'this' and return it
    /// \param minuend
    /// \param subtrahend
    /// \return 'this'
    ///
    Array<T>& Sub(const Array<T>& minuend,
                  const Array<T>& subtrahend);

    ///
    /// \brief SubAcc subtract the contents of subtrahend from 'this' and return it
    /// \param subtrahend
    /// \return 'this'
    ///
    Array<T>& SubAcc(const Array<T>& subtrahend);

    ///
    /// \brief Mul multiply contents of multiplier with 'this' into a new array and return it. Consider using MulAcc() or Mul(multiplier, multiplicand) instead
    /// \param multiplier
    /// \return
    ///
    Array<T> Mul(const Array<T>& multiplier) const;

    ///
    /// \brief Mul multiply the contents of multiplier and multiplicand into 'this' and return it
    /// \param multiplier
    /// \param multiplicand
    /// \return 'this'
    ///
    Array<T>& Mul(const Array<T>& multiplier,
                  const Array<T>& multiplicand);

    ///
    /// \brief MulAcc multiply multiplier into 'this' and return it
    /// \param multiplier
    /// \return 'this'
    ///
    Array<T>& MulAcc(const Array<T>& multiplier);

    ///
    /// \brief Div divide the contents of 'this' by divisor into a new array and return it
    /// \param divisor
    /// \return
    ///
    Array<T> Div(const Array<T>& divisor) const;

    ///
    /// \brief Div divide the contents of dividend by divisor into 'this' and return it
    /// \param dividend
    /// \param divisor
    /// \return
    ///
    Array<T>& Div(const Array<T>& dividend,
                  const Array<T>& divisor);

    ///
    /// \brief DivAcc divide the contents of 'this' by divisor into 'this' and return it
    /// \param dividend
    /// \return
    ///
    Array<T>& DivAcc(const Array<T>& dividend);

    ///
    /// \brief Sqrt computes the square root of each element in this array and returns it in a new array of the same dimension
    /// \return Array<T> containing the element-wise square roots of this array's buffer
    ///
    Array<T> Sqrt();

    ///
    /// \brief SqrtAcc replaces each entry in this Array's buffer with its square root
    /// \return this
    ///
    Array<T>& SqrtAcc();

    ///
    /// \brief CrossProduct computes the dot product between multiplicand and 'this' vectors, size of multiplicand must be the same as size of 'this'
    /// \param multiplicand
    /// \return scalar dot product
    ///
    T DotProduct(const Array<T>& multiplicand) const;

  protected:
    // The following function pointers are bound to one of the *<op>Impl( ) member functions
    // below based on the ProcessorCaps object which determines the CPU's capabilities
    Array<T> (Array<T>::*AddImpl) (const Array<T>&) const;
    Array<T>& (Array<T>::*Add2Impl) (const Array<T>&, const Array<T>&);
    Array<T>& (Array<T>::*AddAccImpl) (const Array<T>&);
    Array<T> (Array<T>::*SubImpl) (const Array<T>&) const;
    Array<T>& (Array<T>::*Sub2Impl) (const Array<T>&, const Array<T>&);
    Array<T>& (Array<T>::*SubAccImpl) (const Array<T>&);
    Array<T> (Array<T>::*MulImpl) (const Array<T>&) const;
    Array<T>& (Array<T>::*Mul2Impl) (const Array<T>&, const Array<T>&);
    Array<T>& (Array<T>::*MulAccImpl) (const Array<T>&);
    Array<T> (Array<T>::*DivImpl) (const Array<T>&) const;
    Array<T>& (Array<T>::*Div2Impl) (const Array<T>&, const Array<T>&);
    Array<T>& (Array<T>::*DivAccImpl) (const Array<T>&);
    Array<T> (Array<T>::*SqrtImpl) ();
    Array<T>& (Array<T>::*SqrtAccImpl) ();
    
    Array<T> Avx2AddImpl(const Array<T>& addend) const;
    Array<T>& Avx2Add2Impl(const Array<T>& augend, const Array<T>& addend);
    Array<T>& Avx2AddAccImpl(const Array<T>& addend);
    Array<T> Avx2SubImpl(const Array<T>& subtrahend) const;
    Array<T>& Avx2Sub2Impl(const Array<T>& minuend, const Array<T>& subtrahend);
    Array<T>& Avx2SubAccImpl(const Array<T>& subtrahend);
    Array<T> Avx2MulImpl(const Array<T>& multiplicand) const;
    Array<T>& Avx2Mul2Impl(const Array<T>& multiplier, const Array<T>& multiplicand);
    Array<T>& Avx2MulAccImpl(const Array<T>& multiplicand);
    Array<T> Avx2DivImpl(const Array<T>& divisor) const;
    Array<T>& Avx2Div2Impl(const Array<T>& dividend, const Array<T>& divisor);
    Array<T>& Avx2DivAccImpl(const Array<T>& divisor);
    Array<T> Avx2SqrtImpl();
    Array<T>& Avx2SqrtAccImpl();

    Array<T> AvxAddImpl(const Array<T>& addend) const;
    Array<T>& AvxAdd2Impl(const Array<T>& augend, const Array<T>& addend);
    Array<T>& AvxAddAccImpl(const Array<T>& addend);
    Array<T> AvxSubImpl(const Array<T>& subtrahend) const;
    Array<T>& AvxSub2Impl(const Array<T>& minuend, const Array<T>& subtrahend);
    Array<T>& AvxSubAccImpl(const Array<T>& subtrahend);
    Array<T> AvxMulImpl(const Array<T>& multiplicand) const;
    Array<T>& AvxMul2Impl(const Array<T>& multiplier, const Array<T>& multiplicand);
    Array<T>& AvxMulAccImpl(const Array<T>& multiplicand);
    Array<T> AvxDivImpl(const Array<T>& divisor) const;
    Array<T>& AvxDiv2Impl(const Array<T>& dividend, const Array<T>& divisor);
    Array<T>& AvxDivAccImpl(const Array<T>& divisor);
    Array<T> AvxSqrtImpl();
    Array<T>& AvxSqrtAccImpl();

    void BuildArchBinding();
    void BuildAvxArchBinding();
    void BuildAvx2ArchBinding();
    void BuildFallbackArchBinding();

    Array<T> FallbackAddImpl(const Array<T>& addend) const
    {
      Array<T> sum(this->_buffer.size());
      for ( size_t i = 0; i < this->_buffer.size(); ++i ) {
        sum[i] = this->_buffer[i] + addend._buffer[i];
      }
      return std::move(sum);
    }

    Array<T>& FallbackAdd2Impl(const Array<T>& augend,
                           const Array<T>& addend)
    {
      for ( size_t i = 0; i < this->_buffer.size(); ++i ) {
        this->_buffer[i] = augend._buffer[i] + addend._buffer[i];
      }
      return *this;
    }

    Array<T>& FallbackAddAccImpl(const Array<T>& addend)
    {
      for ( size_t i = 0; i < this->_buffer.size(); ++i )
        this->_buffer[i] += addend._buffer[i];

      return *this;
    }

    Array<T> FallbackSubImpl(const Array<T>& subtrahend) const
    {
      Array<T> difference(this->_buffer.size());
      for ( size_t i = 0; i < this->_buffer.size(); ++i ) {
        difference._buffer[i] = this->_buffer[i] - subtrahend._buffer[i];
      }
      return std::move(difference);
    }

    Array<T>& FallbackSub2Impl(const Array<T>& minuend,
                           const Array<T>& subtrahend)
    {
      for ( size_t i = 0; i < this->_buffer.size(); ++i ) {
        this->_buffer[i] = minuend._buffer[i] - subtrahend._buffer[i];
      }
      return *this;
    }

    Array<T>& FallbackSubAccImpl(const Array<T>& subtrahend)
    {
      for ( size_t i = 0; i < this->_buffer.size(); ++i )
        this->_buffer[i] -= subtrahend._buffer[i];

      return *this;
    }

    Array<T> FallbackMulImpl(const Array<T>& multiplicand) const
    {
      Array<T> product(this->_buffer.size());
      for ( size_t i = 0; i < this->_buffer.size(); ++i ) {
        product._buffer[i] = this->_buffer[i] * multiplicand._buffer[i];
      }
      return std::move(product);
    }

    Array<T>& FallbackMul2Impl(const Array<T>& multiplier,
                           const Array<T>& multiplicand)
    {
      for ( size_t i = 0; i < this->_buffer.size(); ++i ) {
        this->_buffer[i] = multiplier._buffer[i] * multiplicand._buffer[i];
      }
      return *this;
    }

    Array<T>& FallbackMulAccImpl(const Array<T>& multiplicand)
    {
      for ( size_t i = 0; i < this->_buffer.size(); ++i )
        this->_buffer[i] *= multiplicand._buffer[i];

      return *this;
    }

    Array<T> FallbackDivImpl(const Array<T>& divisor) const
    {
      Array<T> quotient(this->_buffer.size());
      for ( size_t i = 0; i < this->_buffer.size(); ++i ) {
        quotient._buffer[i] = this->_buffer[i] / divisor._buffer[i];
      }
      return std::move(quotient);
    }

    Array<T>& FallbackDiv2Impl(const Array<T>& dividend,
                           const Array<T>& divisor)
    {
      for ( size_t i = 0; i < this->_buffer.size(); ++i ) {
        this->_buffer[i] = dividend._buffer[i] / divisor._buffer[i];
      }
      return *this;
    }

    Array<T>& FallbackDivAccImpl(const Array<T>& divisor)
    {
      for ( size_t i = 0; i < this->_buffer.size(); ++i )
        this->_buffer[i] /= divisor._buffer[i];

      return *this;
    }

    Array<T> FallbackSqrtImpl()
    {
      Array<T> res(this->_buffer.size());
      for ( size_t i = 0; i < this->_buffer.size(); ++i )
        res[i] = sqrt(this->_buffer[i]);

      return std::move(res);
    }

    Array<T>& FallbackSqrtAccImpl()
    {
      for ( size_t i = 0; i < this->_buffer.size(); ++i )
        this->_buffer[i] = sqrt(this->_buffer[i]);

      return *this;
    }
  };
  
  // Developers should use the following types rather than define your own templated types
  typedef Array<sp_t> SinglePrecisionArray;
  typedef Array<dp_t> DoublePrecisionArray;
}
