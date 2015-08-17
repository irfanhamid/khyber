#khyber

Khyber is a C++ library that provides high-performance vector processing
primitives. Under the hood it uses the fastest SIMD instruction set available on
the CPU. Currently the implementation exploits the AVX2, AVX and SSE instruction
sets on x64 processors. Khyber exposes its functionality in the form of a
drop-in replacement for the C++ `std::vector<T>` data type `khyber::Array<T>`.

Currently khyber supports 1D vector types `khyber::Array<T>` for the following
data types:

1. ui32_t: unsigned 32-bit integer type, uint32_t in C++;  
2. ui16_t: unsigned 16-bit integer type, uint16_t in C++;  
3. i32_t: signed 32-bit integer type, int32_t in C++;  
4. i16_t: signed 16-bit integer type, int16_t in C++;  
2. sp_t: single-precision floating point, float in C++.  

khyber::Array<T> provides addition, subtraction, multiplication, division and
some other numerical operations such as square root and reciprocation. Detailed
documentation can be found in the doxygen help for the library.

## Structure

The library is layered with the programmer API present in the khyber
namespace. This is where the `Array<T>` class is defined, which is the primary
object provided by this library.

`khyber: khyber namespace, programmer API including the `Array<T>` class and
helper classes like ProcessorCaps, SimdAllocator and SimdContainer  
   |  
   |--- arch/avx: avx namespace, impl of the optimized vector operations using
   the AVX instruction sets  
   |  
   |--- arch/avx2: avx2 namespace, impl of the optimized vector operations using
   the AVX2 instruction sets`  

## Optimization mechanism

The `Array<T>` class works by building a hand-written vtable of function
pointers for arithmetic functionality such as Add, Sub, Mul, ScalarDiv, Sqrt
etc. These are all wired to the most optimistic processor capability at the time
of construction of the `Array<T>` object. After this wiring, each public
function like Add() is actually a stub that calls the correct underlying impl.

The *underlying impl* mentioned above are contained in the arch/avx and
arch/avx2 directories under these respective namespaces. These contain the files
AvxInternals.cpp and Avx2Internals.cpp which have functions such as
InternalAdd(), InternalSub() etc. These functions are implemented using
optimized assembly via the intrinsics mechanism.


khyber achieves optimum performance by executing dynamic dispatch of the
`Array<T>` operations such as Add, Subtract etc. as a function of the CPU
capabilities on which the code is running. These capabilities are queried and
stored in the ProcessorCaps object. A neat trick to take full advantage of
compiler optimization while keeping safety is that arch/avx is compiled with the
-mavx flag, and arch/avx2 with the -mavx2 flag. This would be unsafe in a
regular program because you could run arch/avx2 code on a processor that doesn't
support it and get an IllegalInstruction exception. However, since these
packages are hidden behind the dynamic dispatch capability of khyber::Array<T>
everything works out fine.