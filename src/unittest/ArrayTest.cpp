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

#include <boost/test/unit_test.hpp>
#include "Array.hpp"

BOOST_AUTO_TEST_SUITE(ArrayTestSuite)

khyber::SinglePrecisionArray MakeArray(khyber::sp_t** pBuffer)
{
  khyber::SinglePrecisionArray tmp;
  *pBuffer = tmp.data();
  return tmp;
}

BOOST_AUTO_TEST_CASE(TestArrayConstructors)
{
  khyber::SinglePrecisionArray arr0(512);
  BOOST_CHECK_EQUAL(arr0.size(), 512);
  BOOST_CHECK_EQUAL((uint64_t)arr0.data() % 32, 0);
  
  for ( int i = 0; i < 512; ++i )
    arr0[i] = i;
  
  khyber::SinglePrecisionArray arr1(arr0);
  BOOST_CHECK_EQUAL(arr1.size(), 512);
  BOOST_CHECK_EQUAL((uint64_t)arr1.data() % 32, 0);
  BOOST_CHECK_NE(arr0.data(), arr1.data());
  for ( int i = 0; i < 512; ++i )
    BOOST_CHECK_EQUAL(arr0.at(i), arr1.at(i));
  
  khyber::sp_t* pBuffer = nullptr;
  khyber::SinglePrecisionArray arr2(MakeArray(&pBuffer));
  BOOST_CHECK_EQUAL(arr2.data(), pBuffer);
  arr2 = arr1;
  BOOST_CHECK_NE(arr1.data(), arr2.data());
  arr2 = MakeArray(&pBuffer);
  BOOST_CHECK_EQUAL(arr2.data(), pBuffer);
}

BOOST_AUTO_TEST_CASE(TestArrayAdd)
{
  khyber::SinglePrecisionArray arr0(512);
  for ( size_t i = 0; i < 512; ++i )
    arr0[i] = i;
  
  khyber::SinglePrecisionArray arr1(arr0);
  for ( size_t i = 0; i < 512; ++i )
    arr1[i] *= 2;
  khyber::SinglePrecisionArray sum;
  sum = arr0.Add(arr1);
  for ( size_t i = 0; i < 512; ++i ) {
    if ( sum[i] != (arr0[i] + arr1[i]) ) {
      BOOST_CHECK_MESSAGE(false, i);
      break;
    }
  }

  // Lets also test an array size not a multiple of 8
  khyber::SinglePrecisionArray arr2(13);
  khyber::SinglePrecisionArray arr3(arr2);
  for ( size_t i = 0; i < 13; ++i ) {
    arr2[i] = 10 * i;
    arr3[i] = 20 * i;
  }
  sum = arr2.Add(arr3);
  for ( size_t i = 0; i < 13; ++i ) {
    if ( sum[i] != (arr2[i] + arr3[i]) ) {
      BOOST_CHECK_MESSAGE(false, i);
      break;
    }
  }
}

BOOST_AUTO_TEST_CASE(TestArraySqrt)
{
  khyber::SinglePrecisionArray arr0(512);
}

BOOST_AUTO_TEST_SUITE_END()
