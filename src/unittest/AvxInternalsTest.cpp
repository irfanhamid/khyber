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
#include "ProcessorCaps.hpp"
#include "AvxInternals.hpp"

#define EPSILON 0.001
#define TEST_VECTOR_LENGTH 512

BOOST_AUTO_TEST_SUITE(AvxInternalsTestSuite)

using namespace khyber;

BOOST_AUTO_TEST_CASE(TestAvxAddSinglePrecision)
{
  ProcessorCaps caps;
  if ( !caps.IsAvx() ) {
    return;
  }

  sp_t sum[TEST_VECTOR_LENGTH];
  sp_t augend[TEST_VECTOR_LENGTH];
  sp_t addend[TEST_VECTOR_LENGTH];
  for ( size_t i = 0; i < TEST_VECTOR_LENGTH; ++i ) {
    augend[i] = i;
    addend[i] = 2 * i;
  }
  
  avx::InternalAdd(TEST_VECTOR_LENGTH,
                   sum,
                   augend,
                   addend);
  for ( size_t i = 0; i < TEST_VECTOR_LENGTH; ++i ) {
    if ( sum[i] != (augend[i] + addend[i]) ) {
      BOOST_CHECK_MESSAGE(false, i);
      break;
    }
  }
}

BOOST_AUTO_TEST_CASE(TestAvxSubSinglePrecision)
{
  ProcessorCaps caps;
  if ( !caps.IsAvx() ) {
    return;
  }

  sp_t difference[TEST_VECTOR_LENGTH];
  sp_t minuend[TEST_VECTOR_LENGTH];
  sp_t subtrahend[TEST_VECTOR_LENGTH];
  for ( size_t i = 0; i < TEST_VECTOR_LENGTH; ++i ) {
    minuend[i] = i * i;
    subtrahend[i] = (i - 1) * (i - 1);
  }

  avx::InternalSub(TEST_VECTOR_LENGTH,
                   difference,
                   minuend,
                   subtrahend);
  for ( size_t i = 0; i < TEST_VECTOR_LENGTH; ++i ) {
    if ( difference[i] != (minuend[i] - subtrahend[i]) ) {
      BOOST_CHECK_MESSAGE(false, i);
      break;
    }
  }
}

BOOST_AUTO_TEST_CASE(TestAvxMulSinglePrecision)
{
  ProcessorCaps caps;
  if ( !caps.IsAvx() ) {
    return;
  }

  sp_t product[TEST_VECTOR_LENGTH];
  sp_t multiplier[TEST_VECTOR_LENGTH];
  sp_t multiplicand[TEST_VECTOR_LENGTH];
  for ( size_t i = 0; i < TEST_VECTOR_LENGTH; ++i ) {
    multiplier[i] = i;
    multiplicand[i] = (i % 2 ? -1 : 1) * i;
  }

  avx::InternalMul(TEST_VECTOR_LENGTH,
                   product,
                   multiplier,
                   multiplicand);
  for ( auto i = 0; i < TEST_VECTOR_LENGTH; ++i ) {
    if ( product[i] != multiplier[i] * multiplicand[i] ) {
      BOOST_CHECK_MESSAGE(false, i);
      break;
    }
  }
}

BOOST_AUTO_TEST_CASE(TestAvxDivSinglePrecision)
{
  ProcessorCaps caps;
  if ( !caps.IsAvx() ) {
    return;
  }

  sp_t quotient[TEST_VECTOR_LENGTH];
  sp_t dividend[TEST_VECTOR_LENGTH];
  sp_t divisor[TEST_VECTOR_LENGTH];
  for ( auto i = 0; i < TEST_VECTOR_LENGTH; ++i ) {
    dividend[i] = i;
    divisor[i] = 1 + ((i % 2 ? -1 : 1) * i * i);
  }

  avx::InternalDiv(TEST_VECTOR_LENGTH,
                   quotient,
                   dividend,
                   divisor);
  for ( auto i = 0; i < TEST_VECTOR_LENGTH; ++i ) {
    if ( quotient[i] != dividend[i] / divisor[i] ) {
      BOOST_CHECK_MESSAGE(false, i);
      break;
    }
  }
}

BOOST_AUTO_TEST_CASE(TestAvxSqrtSinglePrecision)
{
  ProcessorCaps caps;
  if ( !caps.IsAvx() ) {
    return;
  }

  sp_t src[TEST_VECTOR_LENGTH];
  sp_t dst[TEST_VECTOR_LENGTH];
  for ( auto i = 0; i < TEST_VECTOR_LENGTH; ++i ) {
    src[i] = i * (i + 10);
  }

  avx::InternalSqrt(TEST_VECTOR_LENGTH,
                    dst,
                    src);
  for ( auto i = 0; i < TEST_VECTOR_LENGTH; ++i ) {
    if ( abs(dst[i] * dst[i] - src[i]) > EPSILON ) {
      BOOST_CHECK_MESSAGE(false, i);
      break;
    }
  }
}

BOOST_AUTO_TEST_CASE(TestAvxDotProductSinglePrecision)
{
  ProcessorCaps caps;
  if ( !caps.IsAvx() ) {
    return;
  }

  sp_t product;
  sp_t multiplicand[TEST_VECTOR_LENGTH];
  sp_t multiplier[TEST_VECTOR_LENGTH];

  sp_t refVal = 0;
  for ( auto i = 0; i < TEST_VECTOR_LENGTH; ++i ) {
    multiplicand[i] = 0.1 * i;
    multiplier[i] = i;
    refVal += (multiplicand[i] * multiplier[i]);
  }

  avx::InternalDotProduct(TEST_VECTOR_LENGTH,
                          &product,
                          multiplicand,
                          multiplier);
  BOOST_CHECK_EQUAL(refVal, product);
}

BOOST_AUTO_TEST_SUITE_END()
