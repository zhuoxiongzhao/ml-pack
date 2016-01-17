AR=ar rc
RANLIB=ranlib
CXX=g++
CC=gcc
CPPFLAGS=
CFLAGS=-g -Wall -O2 -I.
CXXFLAGS=-g -Wall -O2 -I.
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

OBJECT=common/daxpy.o \
common/dcopy.o \
common/ddot.o \
common/dscal.o \
common/dswap.o \
common/city.o \
common/hash-entry.o \
common/hash-function.o \
common/lbfgs.o \
common/mt19937ar.o \
common/mt19937-64.o \
lda/alias.o \
lda/rand.o \
lda/sampler.o \
lr/lr.o \
lr/metric.o \
lr/problem.o

LIB=libx.a

BIN= \
lda-test$(EXE) \
lr-main$(EXE) \
lr-test$(EXE) \
gen-feature-map$(EXE) \
map-sample$(EXE) \
problem-gen-bin$(EXE) \
problem-load-bin$(EXE)

.PHONY: clean
all: $(LIB) $(BIN)

$(LIB): $(OBJECT)
	$(AR) $@ $^
	$(RANLIB) $@

common/daxpy.o: common/daxpy.c
	$(CC) $(CFLAGS) -c -o $@ $<

common/dcopy.o: common/dcopy.c
	$(CC) $(CFLAGS) -c -o $@ $<

common/ddot.o: common/ddot.c
	$(CC) $(CFLAGS) -c -o $@ $<

common/dscal.o: common/dscal.c
	$(CC) $(CFLAGS) -c -o $@ $<

common/dswap.o: common/dswap.c
	$(CC) $(CFLAGS) -c -o $@ $<

common/lbfgs.o: common/lbfgs.c
	$(CC) $(CFLAGS) -c -o $@ $<

common/hash-function.o: common/hash-function.c
	$(CC) $(CFLAGS) -c -o $@ $<

common/mt19937-64.o: common/mt19937-64.c
	$(CC) $(CFLAGS) -c -o $@ $<

common/mt19937ar.o: common/mt19937ar.c
	$(CC) $(CFLAGS) -c -o $@ $<

lda-test$(EXE): lda/lda-test.cc $(LIB)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

lr-main$(EXE): lr/lr-main.cc $(LIB)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

lr-test$(EXE): lr/lr-test.cc $(LIB)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

gen-feature-map$(EXE): lr/gen-feature-map.cc $(LIB)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

map-sample$(EXE): lr/map-sample.cc $(LIB)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

problem-gen-bin$(EXE): lr/problem-gen-bin.cc $(LIB)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

problem-load-bin$(EXE): lr/problem-load-bin.cc $(LIB)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

clean:
	rm -f $(OBJECT) $(LIB) $(BIN)
