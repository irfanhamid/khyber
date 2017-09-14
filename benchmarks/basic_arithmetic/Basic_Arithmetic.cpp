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

#include <vector>

#include "../BenchmarkApp.hpp"
#include "Array.hpp"
#include "Serial.hpp"

using namespace khyber;

class BasicArithmeticBenchmarks : public BenchmarkApp
{
public:
  virtual bool InitApplication(int argc, char* argv[])
  {
    if ( !BenchmarkApp::InitApplication(argc, argv) ) {
      return false;
    }

    _simdDst.resize(_length);
    _simdSrc.resize(_length);
    _dst.resize(_length);
    _src.resize(_length);

    std::cout << "Length = " << _length << std::endl;

    return true;
  }

  virtual bool RunSimd()
  {
    _simdDst.Add(_simdSrc);

    return true;
  }

  virtual bool RunSerial()
  {
    RunSerialImpl(_dst, _src);
    return true;
  }

private:
  SinglePrecisionArray _simdSrc;
  SinglePrecisionArray _simdDst;

  std::vector<float> _src;
  std::vector<float> _dst;
};

BasicArithmeticBenchmarks theApp;
BenchmarkApp* app = (BenchmarkApp*)&theApp;
