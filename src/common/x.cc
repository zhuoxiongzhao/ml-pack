// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//

#include "x.h"

FILE* yfopen(const char* filename, const char* mode) {
  FILE* fp = fopen(filename, mode);
  if (fp == NULL) {
    Error("Open \"%s\" failed\n", filename);
  }
  return fp;
}

FILE* xfopen(const char* filename, const char* mode) {
  FILE* fp = fopen(filename, mode);
  if (fp == NULL) {
    Error("Open \"%s\" failed\n", filename);
    exit(1);
  }
  return fp;
}

void* xmalloc(size_t size) {
  void* p = malloc(size);
  if (p == NULL) {
    Error("malloc %d bytes failed\n", (int)size);
    exit(1);
  }
  return p;
}

void* xrealloc(void* memory, size_t new_size) {
  void* p = realloc(memory, new_size);
  if (p == NULL) {
    Error("realloc %d bytes failed\n", (int)new_size);
    exit(1);
  }
  return p;
}

double xatof(const char* str) {
  char* endptr;
  double d;
  errno = 0;
  d = strtod(str, &endptr);
  if (errno != 0 || str == endptr) {
    Error("%s is not a double\n", str);
    exit(1);
  }
  return d;
}

int xatoi(const char* str) {
  char* endptr;
  int i;
  errno = 0;
  i = (int)strtol(str, &endptr, 10);
  if (errno != 0 || str == endptr) {
    Error("%s is not an integer\n", str);
    exit(1);
  }
  return i;
}
