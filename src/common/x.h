// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// common utils
//

#ifndef SRC_COMMON_X_H_
#define SRC_COMMON_X_H_

#include <errno.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define Malloc(type, n) (type*)xmalloc(((size_t)(n))*sizeof(type))

#if defined _NDEBUG || defined NDEBUG
#define Debug(...)
#else
#define Debug(...) do {fprintf(stderr, "[Debug]");\
    fprintf(stderr, __VA_ARGS__);} while (0)
#endif
#define Log(...) do {fprintf(stderr, "[Log]");\
    fprintf(stderr, __VA_ARGS__);} while (0)
#define Error(...) do {fprintf(stderr, "[Error]");\
    fprintf(stderr, __VA_ARGS__);} while (0)

FILE* yfopen(const char* filename, const char* mode);
FILE* xfopen(const char* filename, const char* mode);
void* xmalloc(size_t size);
void* xrealloc(void* memory, size_t new_size);
double xatof(const char* str);
int xatoi(const char* str);

class ScopedFile {
 private:
  FILE* px_;
  ScopedFile(const ScopedFile&);
  ScopedFile& operator=(const ScopedFile&);

 public:
  explicit ScopedFile(FILE* p = NULL) : px_(p) {}

  ~ScopedFile() {
    if (px_) {
      fclose(px_);
    }
  }
};

template <class T>
class ScopedPtrMalloc {
 private:
  T ptr_;
  ScopedPtrMalloc(const ScopedPtrMalloc&);
  ScopedPtrMalloc& operator=(const ScopedPtrMalloc&);

 public:
  explicit ScopedPtrMalloc(T p = NULL): ptr_(p) {}

  ~ScopedPtrMalloc() {
    if (ptr_) {
      free((void*)ptr_);
    }
  }

  void set_for_realloc(T p) {
    ptr_ = p;
  }
};

#endif  // SRC_COMMON_X_H_
