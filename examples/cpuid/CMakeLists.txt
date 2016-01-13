cmake_minimum_required(VERSION 2.8.7)

set(CMAKE_CXX_FLAGS "-O3 -std=c++11")
include_directories("../../src/lib")
project(cpuid)
add_executable(cpuid Cpuid.cpp)
