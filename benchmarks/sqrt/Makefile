CC=g++

OBJS=Sqrt.o \

LIBS=-lboost_program_options \
	-lkhyber \

COMPNAME=sqrt_benchmarks

CXXFLAGS=-mavx -mavx2 -msse -msse2 -msse3 -msse4 -mfma -std=c++11 -O3

default: ${COMPNAME}

${COMPNAME}: ${OBJS}
	${CC} ${CXXFLAGS} -o ${COMPNAME} ${OBJS} ${LIBS} -L../../src/lib/

%.o: %.cpp
	${CC} ${CXXFLAGS} -c -o $@ $< -I../../src/lib/

clean:
	rm -rf *~ *.o *.so *.a ${COMPNAME}
