#khyber

Library for easy usage of the Intel64 vector processing instruction
sets (SSE, AVX).

## Structure

This library is divided into logically different packages with the
programmer API exposed at the top-level `khyber` namespace. This
namespace contains the Array<T> templated type which can be
instantiated with the following types:

(1) ui32_t: unsigned 32-bit integer type, corresponding to the uint32_t in C++;
(2) sp_t: single-precision floating point, corresponding to the float in C++.

The khyber namespace contains the aforementioned Array class as well
as some helper classes like ProcessorCaps, SimdAllocator and
SimdContainer. The Array<T> class works by building a hand-written
vtable of function pointers for arithmetic functionality such as Add,
Sub, Mul, ScalarDiv, Sqrt etc. These are all wired to the most
optimistic processor capability at the time of construction of the
Array<T> object. After this wiring, each public function like Add( )
is actually a stub that calls the correct underlying impl.

The *underlying impl* mentioned above are contained in the arch/avx
and arch/avx2 directories under these respective namespaces. These
contain the files AvxInternals.cpp and Avx2Internals.cpp which have
functions such as InternalAdd(), InternalSub() etc. These functions
are implemented using optimized assembly via the intrinsics
mechanism.

## Optimization mechanism

khyber achieves optimum performance by executing dynamic dispatch of
the Array<T> operations such as Add, Subtract etc. as a function of
the CPU capabilities on which the code is running. These capabilities
are queried and stored in the ProcessorCaps object. A neat trick to
take full advantage of compiler optimization while keeping safety is
that arch/avx is compiled with the -mavx flag, and arch/avx2 with the
-mavx2 flag. This would be unsafe in a regular program because you
could run arch/avx2 code on a processor that doesn't support it and
get an IllegalInstruction exception. However, since these packages are
hidden behind the dynamic dispatch capability of khyber::Array<T>
everything works out fine.