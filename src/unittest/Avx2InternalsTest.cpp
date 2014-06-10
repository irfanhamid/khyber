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
#include "Avx2Internals.hpp"

#define TEST_VECTOR_LENGTH 512

BOOST_AUTO_TEST_SUITE(Avx2InternalsTestSuite)

BOOST_AUTO_TEST_CASE(TestAvx2AddSinglePrecision)
{
  khyber::sp_t sum[TEST_VECTOR_LENGTH];
  khyber::sp_t addend0[TEST_VECTOR_LENGTH];
  khyber::sp_t addend1[TEST_VECTOR_LENGTH];
  for ( size_t i = 0; i < TEST_VECTOR_LENGTH; ++i ) {
    addend0[i] = i;
    addend1[i] = 2 * i;
  }

  khyber::avx2::InternalAdd(TEST_VECTOR_LENGTH,
                           sum,
                           addend0,
                           addend1);
  for ( size_t i = 0; i < TEST_VECTOR_LENGTH; ++i ) {
    if ( sum[i] != (addend0[i] + addend1[i]) ) {
      BOOST_CHECK_MESSAGE(false, i);
      break;
    }
  }
}

BOOST_AUTO_TEST_CASE(TestAvx2DotProductSinglePrecision)
{
  khyber::sp_t product;
  khyber::sp_t multiplicand[TEST_VECTOR_LENGTH];
  khyber::sp_t multiplier[TEST_VECTOR_LENGTH];

  khyber::sp_t refVal = 0;
  for ( size_t i = 0; i < TEST_VECTOR_LENGTH; ++i ) {
    multiplicand[i] = 0.1 * i;
    multiplier[i] = i;
    refVal += (multiplicand[i] * multiplier[i]);
  }

  khyber::avx2::InternalDotProduct(TEST_VECTOR_LENGTH,
                                   &product,
                                   multiplicand,
                                   multiplier);
  BOOST_CHECK_EQUAL(refVal, product);
}

BOOST_AUTO_TEST_SUITE_END()
