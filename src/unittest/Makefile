CC=clang++

OBJS=KhyberTest.o \
	SimdContainerTest.o \
	AvxInternalsTest.o \
	Avx2InternalsTest.o \
	ArrayTest.o \

LIBS=-lboost_unit_test_framework \
	-lkhyber \

COMPNAME=khyber_unittest

CXXFLAGS=-mavx -mavx2 -msse -msse2 -msse3 -msse4 -mfma -std=c++11 -O3

default: ${COMPNAME}

${COMPNAME}: ${OBJS}
	${CC} ${CXXFLAGS} -o ${COMPNAME} ${OBJS} ${LIBS} -L../lib/

%.o: %.cpp
	${CC} ${CXXFLAGS} -c -o $@ $< -I../lib/

clean:
	rm -rf *~ *.o *.so *.a ${COMPNAME}
