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
  ///
  /// \brief The base class for all SIMD data structures, encapsulates an aligned std::vector<T> alongwith processor information.
  ///
  template<typename T>
  class SimdContainer
  {
  public:
    ///
    /// \brief vector_type the underlying buffer's type, i.e., std::vector<T, ...>
    ///
    typedef std::vector<T, SimdAllocator<T, DEFAULT_ALIGNMENT>> vector_type;
    
    ///
    /// \brief SimdContainer Construct a container of default size
    ///
    SimdContainer()
    {
      _buffer.resize(DEFAULT_CONTAINER_SIZE);
    }

    ///
    /// \brief SimdContainer Construct a container of specified size
    /// \param capacity the initial capacity of the container
    ///
    SimdContainer(size_t capacity)
    {
      _buffer.resize(capacity);
    }
    
    ///
    /// \brief SimdContainer const copy constructor
    /// \param rhs the SimdContainer<T> to copy from
    ///
    SimdContainer(const SimdContainer<T>& rhs)
    {
      _buffer.assign(rhs._buffer.begin(), rhs._buffer.end());
    }
    
    ///
    /// \brief SimdContainer non-const copy constructor
    /// \param rhs the SimdContainer<T> to copy from
    ///
    SimdContainer(SimdContainer<T>& rhs)
    {
      _buffer.assign(rhs._buffer.begin(), rhs._buffer.end());
    }

    ///
    /// \brief SimdContainer non-const move constructor
    /// \param rhs the SimdContainer<T> to pilfer from
    ///
    SimdContainer(SimdContainer<T>&& rhs) : _buffer(std::move(rhs._buffer))
    {
    }
    
    ///
    /// \brief operator = move assignment operator
    /// \param rhs the SimdContainer<T> to pilfer from
    /// \return 'this'
    ///
    SimdContainer<T>& operator = (SimdContainer<T>&& rhs)
    {
      _buffer = std::move(rhs._buffer);
      return *this;
    }

    ///
    /// \brief operator = copy assignment operator
    /// \param rhs the SimdContainer<T> to copy from
    /// \return 'this'
    ///
    SimdContainer<T>& operator = (SimdContainer<T>& rhs)
    {
      _buffer.assign(rhs._buffer.begin(), rhs._buffer.end());
      return *this;
    }

    ///
    /// \brief push_back Adds a new element at the end of the vector, after its current last element. The content of val is copied (or moved) to the new element
    /// \param val Value to be copied (or moved) to the new element
    ///
    void push_back(const T& val)
    {
      _buffer.push_back(val);
    }

    ///
    /// \brief push_back Adds a new element at the end of the vector, after its current last element. The content of val is copied (or moved) to the new element
    /// \param val Value to be copied (or moved) to the new element
    ///
    void push_back(T&& val)
    {
      _buffer.push_back(val);
    }

    ///
    /// \brief size returns the number of elements of type T in the underlying memory buffer
    /// \return number of elements
    ///
    size_t size() const
    {
      return _buffer.size();
    }

    ///
    /// \brief capacity the capacity of the underlying memory buffer
    /// \return capacity
    ///
    size_t capacity() const
    {
      return _buffer.capacity();
    }
    
    ///
    /// \brief data return the mutable pointer to the start of the underlying memory buffer, use with extreme care
    /// \return non-const pointer to data
    ///
    T* data()
    {
      return _buffer.data();
    }

    ///
    /// \brief data return the const pointer to the start of the underlying memory buffer, use with extreme care
    /// \return const pointer to data
    ///
    const T* data() const
    {
      return _buffer.data();
    }

    ///
    /// \brief swap take the given buffer as this container's underlying buffer, it will be deleted when this container is destroyed
    /// \param buffer the SimdContainer<T> whose contents to take
    ///
    void swap(vector_type& buffer)
    {
      _buffer.swap(buffer);
    }

  protected:
    ProcessorCaps _procCaps;
    vector_type _buffer;
  };
}
