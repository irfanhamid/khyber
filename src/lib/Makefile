CC=g++

SRCS=Array.cpp \
	AvxInternals.cpp \
	Avx2Internals.cpp \

OBJS=Array.o \
  AvxInternals.o \
  Avx2Internals.o \

COMPNAME=libkhyber

CXXFLAGS=-mavx -mavx2 -msse -msse2 -msse3 -msse4 -mfma -std=c++11 -g

default: ${COMPNAME}.a

${COMPNAME}.a: ${OBJS}
	libtool -static -o ${COMPNAME}.a *.o

%.o: %.cpp
	${CC} ${CXXFLAGS} -c -o $@ $<

clean:
	rm -rf *~ *.o *.so *.a
