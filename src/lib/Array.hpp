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
    /// Add contents of "this" and addend arrays into a new array and return it.
    /// Consider using AddAcc() instead.
    ///
    Array<T> Add(const Array<T>& addend) const;
    
    ///
    /// Add contents of addend into "this" and return it.
    ///
    Array<T>& AddAcc(const Array<T>& addend);
    
  protected:
    // The following two function pointers are bound to one of the *<op>Impl( ) member functions
    // below based on the ProcessorCaps object which determines the CPU's capabilities
    Array<T> (Array<T>::*AddImpl) (const Array<T>&) const;
    Array<T>& (Array<T>::*AddAccImpl) (const Array<T>&);
    void BuildArchBinding();
    
    Array<T> Avx2AddImpl(const Array<T>& addend) const;
    Array<T>& Avx2AddAccImpl(const Array<T>& addend);
    
    Array<T> AvxAddImpl(const Array<T>& addend) const;
    Array<T>& AvxAddAccImpl(const Array<T>& addend);
    
    Array<T> BaseAddImpl(const Array<T>& addend) const
    {
      Array<T> sum(this->_buffer.size());
      for ( size_t i = 0; i < this->_buffer.size(); ++i ) {
        sum[i] = this->_buffer[i] + addend._buffer[i];
      }
      return std::move(sum);
    }

    Array<T>& BaseAddAccImpl(const Array<T>& addend)
    {
      for ( size_t i = 0; i < this->_buffer.size(); ++i )
        this->_buffer[i] += addend._buffer[i];

      return *this;
    }
  };
  
  // Developers should use the following types rather than define your own templated types
  typedef Array<sp_t> SinglePrecisionArray;
  typedef Array<dp_t> DoublePrecisionArray;
}
