#khyber

Khyber is a C++ library that provides high-performance vector processing
primitives. Under the hood it uses the fastest SIMD instruction set available on
the CPU. Currently the implementation exploits the AVX2, AVX and SSE instruction
sets on x64 processors. Khyber exposes its functionality in the form of a
drop-in replacement for the C++ `std::vector<T>` data type called
`khyber::Array<T>`.

Currently khyber supports 1D vector types `khyber::Array<T>` for the following
data types:

1. ui32_t: unsigned 32-bit integer type, uint32_t in C++;  
2. ui16_t: unsigned 16-bit integer type, uint16_t in C++;  
3. i32_t: signed 32-bit integer type, int32_t in C++;  
4. i16_t: signed 16-bit integer type, int16_t in C++;  
2. sp_t: single-precision floating point, float in C++.  

`khyber::Array<T>` provides addition, subtraction, multiplication, division and
some other numerical operations such as square root and reciprocation. Detailed
documentation of operations provided can be found in the doxygen help for the
library.

## Building

khyber uses cmake to configure and create make files for each
architecture. Currently khyber works on Linux and Mac OSX. It requires clang
LLVM 3.6.0 or gcc 4.8 at a minimum due to its use of C++11 features like rvalue
references. In order to build `libkhyber.a` please git clone the master
repository and invoke `cmake .` from the root khyber directory. In case you have
the compiler aliased to some name you can pass that via CMAKE_CXX_COMPILER:

`cmake . -DCMAKE_CXX_COMPILER=g++-4.8`

Once your Makefiles are generated, you can invoke `make` from any location and
build all child directories. `libkhyber.a` will be generated in `khyber/src/lib/`.

## Structure

The library is layered with the programmer API at the top, present in the khyber
namespace. This is where the `Array<T>` class is defined, which is the primary
user object provided by this library.

`src/lib: programmer API Array<T>, ProcessorCaps, SimdAllocator and SimdContainer`  
`   |`  
`   |--- arch/avx: optimized vector operations with AVX instruction sets`  
`   |`  
`   |--- arch/avx2: optimized vector operations with the AVX2 instruction sets`  
`src/unittest: unittests for the library`  
`benchmarks: performance benchmarks for the khyber library comparing to stock
C++ generated code`
`scratchpad: experimentation code location`  

## Optimization mechanism

The `Array<T>` class works by building a kind of hand-written vtable each time
an object of its type is constructed. The public functions like
`Array<T>::Add()` are actually stubs that invoke function pointers for the
implementation. This is the central idea of khyber: at `Array<T>` object
construction time the CPU is queried for its functionality, and based on that
the most efficient implementation is *wired up* to each of the vtable function
pointers. Thus on a Sandy Bridge and Ivy Bridge CPUs the constructor will wire
most functions to the arch/avx implementations because AVX is the highest
instruction set supported; whereas on newer Haswell CPUs the AVX2 dispatch will
be selected in the arch/avx2 location.

The code in the arch/avx and arch/avx2 locations is implemented using the
high-level assembly constructs provided in modern compilers called
*intrinsics*. These are in effect assembler directives but with automatic
register selection. A neat trick to take full advantage of compiler optimization
while keeping safety is that arch/avx is compiled with the -mavx flag, and
arch/avx2 with the -mavx2 flag. This would be unsafe in a regular program
because you could run arch/avx2 code on a processor that doesn't support it and
get an IllegalInstruction exception. However, since these packages are hidden
behind the dynamic dispatch capability of `Array<T>` everything works out
fine. Preliminary benchmarks show promising performance gains of up to 4x over
stock C++ code even with full optimization turned on.