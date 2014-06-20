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

#include <cmath>
#include <boost/test/unit_test.hpp>
#include "ProcessorCaps.hpp"
#include "Avx2Internals.hpp"

#define EPSILON 0.001
#define TEST_VECTOR_LENGTH 515

BOOST_AUTO_TEST_SUITE(Avx2InternalsTestSuite)

using namespace khyber;

BOOST_AUTO_TEST_CASE(TestAvx2Negate)
{
  ProcessorCaps caps;
  if ( !caps.IsAvx2() ) {
    return;
  }

  sp_t negated[TEST_VECTOR_LENGTH];
  sp_t src[TEST_VECTOR_LENGTH];

  for ( auto i = 0; i < TEST_VECTOR_LENGTH; ++i ) {
    src[i] = i / 10;
  }

  avx2::InternalNegate(TEST_VECTOR_LENGTH,
                       negated,
                       src);
  for ( auto i = 0; i < TEST_VECTOR_LENGTH; ++i ) {
    if ( negated[i] != -src[i] ) {
      BOOST_CHECK_MESSAGE(false, i);
      break;
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
