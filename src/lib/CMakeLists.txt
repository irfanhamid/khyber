cmake_minimum_required(VERSION 2.8.8)

set(CMAKE_CXX_FLAGS "-O3 -std=c++11")
include_directories(".")
set_source_files_properties(arch/avx/AvxInternals.cpp COMPILE_FLAGS "-mavx -mfma")
set_source_files_properties(arch/avx2/Avx2Internals.cpp COMPILE_FLAGS "-mavx2 -mfma")
add_library(khyber Array.cpp arch/avx/AvxInternals.cpp arch/avx2/Avx2Internals.cpp)


