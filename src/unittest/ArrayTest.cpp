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

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ArrayTest
#include <boost/test/unit_test.hpp>

#include "Array.hpp"

BOOST_AUTO_TEST_CASE(TestArrayConstruction)
{
  khyber::SinglePrecisionArray arr0(512);
  BOOST_CHECK(arr0.Size() == 512);
  BOOST_CHECK(arr0.SizeInBytes() == 512 * sizeof(float));
  BOOST_CHECK((uint64_t)arr0.GetBuffer() % 32 == 0);
  
  for ( int i = 0; i < 512; ++i )
  {
    arr0[i] = i;
  }
  khyber::SinglePrecisionArray arr1(arr0);
  
  BOOST_CHECK(arr1.Size() == 512);
  BOOST_CHECK(arr1.SizeInBytes() == 512 * sizeof(float));
  BOOST_CHECK((uint64_t)arr1.GetBuffer() % 32 == 0);
  for ( int i = 0; i < 512; ++i )
  {
    BOOST_CHECK(arr0.At(i) == arr1.At(i));
  }
}

BOOST_AUTO_TEST_CASE(TestArrayAdd)
{
  
}