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
#include "SimdContainer.hpp"

BOOST_AUTO_TEST_SUITE(SimdContainerTestSuite)

khyber::SimdContainer<khyber::sp_t> MakeContainer(khyber::sp_t** addr)
{
  khyber::SimdContainer<khyber::sp_t> temp(512);
  *addr = temp.GetBuffer();
  return std::move(temp);
}

BOOST_AUTO_TEST_CASE(TestSimdContainerConstructors)
{
  khyber::SimdContainer<khyber::sp_t> c0(512);
  BOOST_CHECK(c0.Capacity() == 512);
  for ( size_t i = 0; i < 512; ++i )
    c0.GetBuffer()[i] = i;

  khyber::SimdContainer<khyber::sp_t> c1(c0);
  BOOST_CHECK(c1.Capacity() == 512);
  BOOST_CHECK(c0.GetBuffer() != c1.GetBuffer());
  for ( size_t i = 0; i < 512; ++i )
    BOOST_CHECK(c0.GetBuffer()[i] == c1.GetBuffer()[i]);
  
  khyber::SimdContainer<khyber::sp_t> c2(512);
  c2 = c1;
  BOOST_CHECK(c1.GetBuffer() != c2.GetBuffer());
  for ( size_t i = 0; i < 512; ++i )
    BOOST_CHECK(c1.GetBuffer()[i] == c2.GetBuffer()[i]);
  
  khyber::sp_t* pBuffer;
  c2 = MakeContainer(&pBuffer);
  BOOST_CHECK(pBuffer == c2.GetBuffer());
}

BOOST_AUTO_TEST_SUITE_END()
