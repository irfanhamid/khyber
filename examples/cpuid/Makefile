CC=g++

SRCS=Cpuid.cpp

OBJS=Cpuid.o

LIBS=-lboost_program_options

INCS=../../src/lib

COMPNAME=cpuid

#CXXFLAGS=-mavx -mavx2 -msse -msse2 -msse3 -msse4 -mfma -std=c++11 -O3
CXXFLAGS=-std=c++11 -O3

default: ${OBJS}
	${CC} ${CXXFLAGS} -o ${COMPNAME} ${OBJS} ${LIBS}

%.o: %.cpp
	${CC} ${CXXFLAGS} -c -o $@ $< -I${INCS}

clean:
	rm -rf *~ *.o *.so
