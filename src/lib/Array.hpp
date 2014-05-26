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

#include <boost/scoped_array.hpp>
#include <boost/function.hpp>

#include "Types.hpp"
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
    Array()
    {
      BuildArchBinding();
    }
    
    Array(size_t capacity) : SimdContainer<T>(capacity)
    {
      BuildArchBinding();
    }
    
    ///
    /// Return the dimension (size) of this array
    ///
    inline size_t Dimension() const
    {
      return this->_size;
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
    Array<T>&& Add(const Array<T>& addend) const;
    
    ///
    /// Add contents of addend into "this" and return it.
    ///
    Array<T>& AddAcc(const Array<T>& addend);
    
  protected:
    boost::function<Array<T>&& (const Array<T>*, const Array<T>&)> AddImpl;
    boost::function<Array<T>& (Array<T>*, const Array<T>&)> AddAccImpl;
    void BuildArchBinding();
    
    Array<T>&& Avx2AddImpl(const Array<T>& addend) const;
    Array<T>& Avx2AddAccImpl(const Array<T>& addent);
    
    Array<T>&& AvxAddImpl(const Array<T>& addend) const;
    Array<T>& AvxAddAccImpl(const Array<T>& addend);
    
    Array<T>&& BaseAddImpl(const Array<T>& addend) const
    {
      Array<T> sum(SimdContainer<T>::_size);
      for ( size_t i = 0; i < this->_size; ++i )
      {
        sum[i] = this->_buffer[i] + addend._buffer[i];
      }
      return std::move(sum);
    }

    Array<T>& BaseAddAccImpl(const Array<T>& addend)
    {
      for ( size_t i = 0; i < this->_size; ++i )
      {
        this->_buffer[i] += addend._buffer[i];
      }
      return *this;
    }
  };
  
  // Developers should use the following types rather than define your own templated types
  typedef Array<sp_t> SinglePrecisionArray;
  typedef Array<dp_t> DoublePrecisionArray;
}