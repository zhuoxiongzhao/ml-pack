CXX=g++
CC=gcc
CFLAGS=-Wall -O2
CXXFLAGS=-Wall -O2
LDFLAGS=
SYS=$(shell gcc -dumpmachine)

ifeq (, $(findstring mingw, $(SYS)))
	CFLAGS+=-fPIC
	CXXFLAGS+=-fPIC
else
	LDFLAGS+=-static
endif

.PHONY: blas common lbfgs clean
all: blas common lbfgs

blas: blas/daxpy.o blas/dcopy.o blas/ddot.o blas/dscal.o blas/dswap.o

common: common/problem.o common/problem-load-text.o common/x.o problem-load-text

lbfgs: lbfgs/lbfgs.o lbfgs/lbfgs-test.o lbfgs-test blas

problem-load-text: common/problem-load-text.o common/problem.o common/x.o
	$(CXX) $(LDFLAGS) -o $@ $^

lbfgs-test: lbfgs/lbfgs-test.o lbfgs/lbfgs.o blas/daxpy.o blas/dcopy.o blas/ddot.o blas/dscal.o blas/dswap.o
	$(CXX) $(LDFLAGS) -o $@ $^

clean:
	rm -f blas/*.o common/*.o problem-load-text lbfgs-test
