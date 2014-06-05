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

#include <stdlib.h>
#include <vector>
#include "Types.hpp"
#include "ProcessorCaps.hpp"
#include "SimdAllocator.hpp"

#define DEFAULT_CONTAINER_SIZE 512

namespace khyber
{
  template<typename T>
  class SimdContainer
  {
  public:
    typedef T value_type;
    
    ///
    /// Construct a container of default size
    ///
    SimdContainer()
    {
      _buffer.resize(DEFAULT_CONTAINER_SIZE);
    }

    ///
    /// Construct a container of specified size
    ///
    SimdContainer(size_t capacity)
    {
      _buffer.resize(capacity);
    }
    
    ///
    /// Const copy constructor
    ///
    SimdContainer(const SimdContainer<T>& rhs)
    {
      _buffer.assign(rhs._buffer.begin(), rhs._buffer.end());
    }
    
    ///
    /// Copy constructor
    ///
    SimdContainer(SimdContainer<T>& rhs)
    {
      _buffer.assign(rhs._buffer.begin(), rhs._buffer.end());
    }
    
    ///
    /// Move constructor
    ///
    SimdContainer(SimdContainer<T>&& rhs) : _buffer(std::move(rhs._buffer))
    {
    }
    
    ///
    /// Move assignment operator
    ///
    SimdContainer<T>& operator = (SimdContainer<T>&& rhs)
    {
      _buffer = std::move(rhs._buffer);
      return *this;
    }
    
    ///
    /// Copy assignment operator
    ///
    SimdContainer<T>& operator = (SimdContainer<T>& rhs)
    {
      _buffer.assign(rhs._buffer.begin(), rhs._buffer.end());
      return *this;
    }
    
    ///
    /// Returns the total size of the underlying container
    ///
    size_t Size() const
    {
      return _buffer.size();
    }

    ///
    /// Returns the capacity of the underlying container
    ///
    size_t Capacity() const
    {
      return _buffer.capacity();
    }
             
    ///
    /// Returns the total size of the underlying container in bytes
    ///
    size_t SizeInBytes() const
    {
      return _buffer.size() * sizeof(T);
    }
    
    ///
    /// Get the underlying buffer. Use this method with extreme care
    ///
    T* GetBuffer()
    {
      return _buffer.data();
    }
    
  protected:
    ProcessorCaps _procCaps;
    std::vector<T, SimdAllocator<T, DEFAULT_ALIGNMENT>> _buffer;
  };
}
