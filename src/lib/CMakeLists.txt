cmake_minimum_required(VERSION 2.8.8)

add_library(khyber Array.cpp arch/avx/AvxInternals.cpp arch/avx2/Avx2Internals.cpp)
set_source_files_properties(arch/avx/AvxInternals.cpp COMPILE_DEFINITIONS "-mavx -mfma")
set_source_files_properties(arch/avx2/Avx2Internals.cpp COMPILE_DEFINITIONS "-mavx2 -mfma")
