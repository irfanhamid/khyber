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
#include "Types.hpp"
#include "ProcessorCaps.hpp"

#define DEFAULT_CONTAINER_SIZE 512

namespace khyber
{
  template<typename T>
  class SimdContainer
  {
  public:
    typedef T value_type;
    
    virtual ~SimdContainer()
    {
      if ( _buffer )
        free(_buffer);
    }
    
    ///
    /// Construct a container of default size
    ///
    SimdContainer() : _buffer(NULL)
    {
      this->Allocate(DEFAULT_CONTAINER_SIZE);
      memset(_buffer, 0, this->SizeInBytes());
    }
    
    ///
    /// Construct a container of specified size
    ///
    SimdContainer(size_t capacity) : _buffer(NULL)
    {
      this->Allocate(capacity);
      memset(_buffer, 0, this->SizeInBytes());
    }
    
    ///
    /// Const copy constructor
    ///
    SimdContainer(const SimdContainer<T>& rhs) : _buffer(NULL)
    {
      this->Allocate(rhs.Size());
      memcpy(_buffer, rhs._buffer, rhs.SizeInBytes());
    }
    
    ///
    /// Copy constructor
    ///
    SimdContainer(SimdContainer<T>& rhs) : _buffer(NULL)
    {
      this->Allocate(rhs.Size());
      memcpy(_buffer, rhs._buffer, rhs.SizeInBytes());
    }
    
    ///
    /// Move constructor
    ///
    SimdContainer(SimdContainer<T>&& rhs) : _buffer(nullptr)
    {
      Reset(rhs._buffer, rhs._size);
      rhs._buffer = NULL;
    }
    
    ///
    /// Move assignment operator
    ///
    SimdContainer<T>& operator = (SimdContainer<T>&& rhs)
    {
      Reset(rhs._buffer, rhs._size);
      rhs._buffer = NULL;
      return *this;
    }
    
    ///
    /// Copy assignment operator
    ///
    SimdContainer<T>& operator = (SimdContainer<T>& rhs)
    {
      this->Allocate(rhs.Size());
      memcpy(_buffer, rhs._buffer, rhs.SizeInBytes());
      return *this;
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
    
    ///
    /// Get the underlying buffer. Use this method with extreme care
    ///
    T* GetBuffer() const
    {
      return _buffer;
    }
    
  protected:
    ProcessorCaps _procCaps;
    T*            _buffer;
    size_t        _size;
    
    void Allocate(size_t capacity)
    {
      if ( _buffer )
        free(_buffer);

      posix_memalign((void**)&_buffer, DEFAULT_ALIGNMENT, capacity * sizeof(T));
      _size = capacity;
    }
    
    void Reset(T* buffer, size_t capacity)
    {
      if ( _buffer )
        free(_buffer);

      _buffer = buffer;
      _size = capacity;
    }
  };
}
