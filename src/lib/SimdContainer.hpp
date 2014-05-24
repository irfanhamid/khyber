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
#include <boost/scoped_array.hpp>
#include "ProcessorCaps.hpp"

#define DEFAULT_CONTAINER_SIZE

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
      this->Allocate(DEFAULT_CONTAINER_SIZE);
      memset(_buffer.get(), 0, this->SizeInBytes());
    }
    
    ///
    /// Construct a container of specified size
    ///
    SimdContainer(size_t capacity)
    {
      this->Allocate(capacity);
      memset(_buffer.get(), 0, this->SizeInBytes());
    }
    
    ///
    /// Initialize contents from given container
    ///
    SimdContainer(const SimdContainer<T>& rhs)
    {
      this->Allocate(rhs.Size());
      memcpy(_buffer.get(), rhs._buffer.get(), rhs.SizeInBytes());
    }
    
    ///
    /// Move constructor from another SimdContainer
    ///
    SimdContainer(SimdContainer<T>&& rhs) : _size(rhs._size)
    {
      _buffer.set(rhs._buffer);
      rhs._size = 0;
      rhs._buffer.set(NULL);
    }
    
    ///
    /// Returns the total size of the underlying container
    ///
    size_t Size() const
    {
      return _size;
    }
             
    ///
    /// Returns the total size of the underlying container in bytes
    ///
    size_t SizeInBytes() const
    {
      return _size * sizeof(T);
    }
    
  protected:
    static ProcessorCaps    _procCaps;

    boost::scoped_array<T>  _buffer;
    size_t                  _size;
    
    void Allocate(size_t capacity)
    {
      T* tmpBuffer;
      posix_memalign((void**)&tmpBuffer, DEFAULT_ALIGNMENT, capacity * sizeof(T));
      _buffer.reset(tmpBuffer);
      _size = capacity;
    }
  };
}
