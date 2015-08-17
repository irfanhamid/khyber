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

#include <chrono>
#include <iostream>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

enum AccelerationType { Serial, AVX, AVX2, Optimum };

namespace po = boost::program_options;

class BenchmarkApp
{
public:
  virtual bool InitApplication(int argc, char* argv[])
  {
    _name.assign(argv[0], strlen(argv[0]));

    std::string acceleration;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("length,l", po::value<size_t>(&_length)->default_value(512), "Length of vector")
        ("iterations,it", po::value<size_t>(&_iterations)->default_value(5000000), "Number of iterations")
        ("fma", po::value<bool>(&_useFma)->default_value(false), "Use the FMA instruction")
        ("acceleration,a", po::value<std::string>(&acceleration)->default_value("best"), "Acceleration type to use, AVX, AVX2, Serial etc.");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if ( !AccelerationFromString(acceleration, _acceleration) ) {
      std::cout << "Invalid acceleration type" << std::endl;
      return false;
    }

    return true;
  }

  virtual bool RunBenchmarks()
  {
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;

    
    start = std::chrono::high_resolution_clock::now();
    for ( auto i = 0; i < _iterations; ++i ) {
      RunSimd();
    }
    end = std::chrono::high_resolution_clock::now();
    _simdDuration = (end - start);

    start = std::chrono::high_resolution_clock::now();
    for ( auto i = 0; i < _iterations; ++i ) {
      RunSerial();
    }
    end = std::chrono::high_resolution_clock::now();
    _serialDuration = (end - start);

    return true;
  }

  virtual bool RunSimd()
  {
    return false;
  }

  virtual bool RunSerial()
  {
    return false;
  }

  ///
  /// \brief Produces a concise report of the execution of both benchmarks
  /// \return std::string containing the report
  ///
  std::string Report()
  {
    std::ostringstream reportStream;
    reportStream << "Benchmarks report for " << _name << std::endl;
    reportStream << "Vector size: " << _length << std::endl;
    reportStream << "Iterations: " << _iterations << std::endl;
    reportStream << "FMA: " << _useFma << std::endl;
    reportStream << "Acceleration type: " << StringFromAcceleration(_acceleration) << std::endl;
    reportStream << "Serial ticks: " << _serialDuration.count() << std::endl;
    reportStream << "Simd ticks:   " << _simdDuration.count() << std::endl;
    // reportStream << "Speedup percentage: " << ((double)(_serialDuration - _simdDuration) * 100.0 / (double)_serialDuration) << std::endl;
    return reportStream.str();
  }

protected:
  ///
  /// \brief Length of the arrays to run the benchmark on, modifiable by the -l cmdline param, defaults to 512
  ///
  size_t _length;

  ///
  /// \brief The number of iterations of the benchmark's basic operation to carry out, modifiable by the -it cmdline param, defaults to 5M
  ///
  size_t _iterations;

  ///
  /// \brief Which hardware level to use (serial, AVX, AVX2), modifiable by the -a cmdline param, defaults to the best available on the processor
  ///
  enum AccelerationType _acceleration;

  ///
  /// \brief Whether to use the FMA instruction, modifiable by the -fma cmdline param, defaults to false
  ///
  bool _useFma;

  std::chrono::high_resolution_clock::duration _serialDuration;
  std::chrono::high_resolution_clock::duration _simdDuration;
  std::string _name;

  std::string StringFromAcceleration(AccelerationType acceleration)
  {
    switch (acceleration) {
    case Serial: return "Serial";
    case AVX: return "AVX";
    case AVX2: return "AVX2";
    case Optimum: return "Optimum";
    }
  }

  bool AccelerationFromString(std::string& acceleration, AccelerationType& accelerationType)
  {
    if ( boost::iequals(acceleration, "best") || boost::iequals(acceleration, "optimum") ) {
      _acceleration = Optimum;
    } else if ( boost::iequals(acceleration, "avx2") ) {
      _acceleration = AVX2;
    } else if ( boost::iequals(acceleration, "avx") ) {
      _acceleration = AVX;
    } else if ( boost::iequals(acceleration, "serial") ) {
      _acceleration = Serial;
    } else {
      return false;
    }

    return true;
  }
};

extern BenchmarkApp* app;

int main(int argc, char* argv[])
{
  if ( !app->InitApplication(argc, argv) ) {
    return 1;
  }

  if ( !app->RunBenchmarks() ) {
    return 1;
  }

  std::cout << std::endl << "--------------------------------------------------------------------------------" << std::endl;
  std::cout << std::endl << app->Report();
  std::cout << std::endl << "--------------------------------------------------------------------------------" << std::endl;

  return 0;
}
