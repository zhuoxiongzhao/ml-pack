AR=ar rc
RANLIB=ranlib
CXX=g++
CC=gcc
CFLAGS=-Wall -O2 -I.
CXXFLAGS=-Wall -O2 -I.
LDFLAGS=
SYS=$(shell gcc -dumpmachine)

ifeq ($(OS),Windows_NT)
	EXE=.exe
else
	EXE=
endif

ifeq (, $(findstring mingw, $(SYS)))
	CFLAGS+=-fPIC
	CXXFLAGS+=-fPIC
else
	LDFLAGS+=-static
endif

OBJECT=blas/daxpy.o \
blas/dcopy.o \
blas/ddot.o \
blas/dscal.o \
blas/dswap.o \
common/metric.o \
common/problem.o \
core/lr.o \
lbfgs/lbfgs.o

LIB=libx.a

BIN= \
lr-test$(EXE) \
lbfgs-test$(EXE) \
gen-feature-map$(EXE) \
map-sample$(EXE) \
problem-gen-bin$(EXE) \
problem-load-bin$(EXE)

.PHONY: clean
all: $(LIB) $(BIN) lbfgs

$(LIB): $(OBJECT)
	$(AR) $@ $^
	$(RANLIB) $@

blas/%.o: blas/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

lbfgs/lbfgs.o: lbfgs/lbfgs.c
	$(CC) $(CFLAGS) -c -o $@ $<

lr-test$(EXE): core/lr-test.cc $(LIB)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

lbfgs-test$(EXE): lbfgs/lbfgs-test.c $(LIB)
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $^

gen-feature-map$(EXE): tools/gen-feature-map.cc $(LIB)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

map-sample$(EXE): tools/map-sample.cc $(LIB)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

problem-gen-bin$(EXE): tools/problem-gen-bin.cc $(LIB)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

problem-load-bin$(EXE): tools/problem-load-bin.cc $(LIB)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

clean:
	rm -f $(OBJECT) $(LIB) $(BIN)
