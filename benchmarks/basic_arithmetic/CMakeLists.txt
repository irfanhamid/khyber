cmake_minimum_required(VERSION 2.8.7)

include_directories("../")
include_directories("../../src/lib/")
link_directories("../../src/lib/")
set(CMAKE_CXX_FLAGS "-O3 -std=c++11")

add_executable(basic_arithmetic Basic_Arithmetic.cpp Serial.cpp)
target_link_libraries(basic_arithmetic khyber)
target_link_libraries(basic_arithmetic boost_program_options)
