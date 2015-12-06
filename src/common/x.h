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

#if defined _WIN32
#define strtoll _strtoi64
#define snprintf _snprintf
#endif

#define _Malloc(type, n) (type*)xmalloc(((size_t)(n))*sizeof(type))
#define _Realloc(p, type, n) (type*)xrealloc(p, ((size_t)(n))*sizeof(type))

#if defined _NDEBUG || defined NDEBUG
#define Debug(...)
#else
#define Debug(...) do {\
    fprintf(stderr, "[Debug]");\
    fprintf(stderr, __VA_ARGS__);\
  } while (0)
#endif

#define Log(...) do {\
    fprintf(stderr, "[Log]");\
    fprintf(stderr, __VA_ARGS__);\
  } while (0)

#define Error(...) do {\
    fprintf(stderr, "[Error]");\
    fprintf(stderr, __VA_ARGS__);\
  } while (0)

#define MISSING_ARG(argc, argv, i) \
  Error("\"%s\" wants a value.\n", argv[i])

#define COMSUME_1_ARG(argc, argv, i) \
  do {\
    for (int j = i; j < argc - 1; j++) {\
      argv[j] = argv[j + 1];\
    }\
    argc -= 1;\
  } while (0)

#define COMSUME_2_ARG(argc, argv, i) \
  do {\
    for (int j = i; j < argc - 2; j++) {\
      argv[j] = argv[j + 2];\
    }\
    argc -= 2;\
  } while (0)

#define DELIMITER " \t|\n"

inline FILE* yfopen(const char* filename, const char* mode) {
  FILE* fp = fopen(filename, mode);
  if (fp == NULL) {
    Error("Open \"%s\" failed.\n", filename);
  }
  return fp;
}

inline FILE* xfopen(const char* filename, const char* mode) {
  FILE* fp = fopen(filename, mode);
  if (fp == NULL) {
    Error("Open \"%s\" failed.\n", filename);
    exit(100);
  }
  return fp;
}

inline void xfwrite(const void* buffer, size_t size, size_t count, FILE* fp) {
  size_t r = fwrite(buffer, size, count, fp);
  if (r != count) {
    Error("Failed to fwrite %d items, actually %d items.\n",
          (int)count, (int)r);
    exit(101);
  }
}

inline void xfread(void* buffer, size_t size, size_t count, FILE* fp) {
  size_t r = fread(buffer, size, count, fp);
  if (r != count) {
    Error("Failed to fread %d items, actually %d items.\n",
          (int)(count), (int)r);
    exit(102);
  }
}

inline void* xmalloc(size_t size) {
  void* p = malloc(size);
  if (p == NULL) {
    Error("malloc %d bytes failed.\n", (int)size);
    exit(120);
  }
  return p;
}

inline void* xrealloc(void* memory, size_t new_size) {
  void* p = realloc(memory, new_size);
  if (p == NULL) {
    Error("realloc %d bytes failed.\n", (int)new_size);
    exit(121);
  }
  return p;
}

inline double xatod(const char* str) {
  char* endptr;
  double d;
  errno = 0;
  d = strtod(str, &endptr);
  if (errno != 0 || str == endptr) {
    Error("%s is not a double.\n", str);
    exit(122);
  }
  return d;
}

inline int xatoi(const char* str) {
  char* endptr;
  int i;
  errno = 0;
  i = (int)strtol(str, &endptr, 10);
  if (errno != 0 || str == endptr) {
    Error("%s is not an integer.\n", str);
    exit(123);
  }
  return i;
}

class ScopedFile {
 private:
  FILE* px_;
  ScopedFile(const ScopedFile&);
  ScopedFile& operator=(const ScopedFile&);

 public:
  enum Mode {
    Read = 0,
    Write,
    ReadBinary,
    WriteBinary,
  };

  ScopedFile() : px_(NULL) {}

  ScopedFile(FILE* p) : px_(p) {}  // NOLINT

  ScopedFile(const char* filename, Mode mode) {
    static const char* mode_map[] = {"r", "w", "rb", "wb"};
    px_ = xfopen(filename, mode_map[mode]);
  }

  ~ScopedFile() {
    Close();
  }

  void Close() {
    if (px_) {
      fclose(px_);
    }
  }

  operator FILE* () {
    return px_;
  }
  operator const FILE* () const {
    return px_;
  }
};

// NOTE: this is only for POD types.
// It uses "malloc" and "realloc", but not "new".
template <class T>
class ScopedPtr {
 private:
  T* ptr_;
  ScopedPtr(const ScopedPtr&);
  ScopedPtr& operator=(const ScopedPtr&);

 public:
  ScopedPtr(): ptr_(NULL) {}

  ScopedPtr(T* p): ptr_(p) {}  // NOLINT

  ScopedPtr& operator=(T* p) {
    Free();
    ptr_ = p;
    return *this;
  }

  ~ScopedPtr() {
    Free();
  }

  void Malloc(size_t size) {
    Free();
    ptr_ = _Malloc(T, size);
  }

  void Realloc(size_t size) {
    ptr_ = _Realloc(ptr_, T, size);
  }

  void Free() {
    if (ptr_) {
      ::free(ptr_);
      ptr_ = NULL;
    }
  }

  operator T* () {
    return ptr_;
  }
  operator const T* () const {
    return ptr_;
  }
  T& operator *() {
    return *ptr_;
  }
  const T& operator *() const {
    return *ptr_;
  }
  T& operator[](int i) {
    return ptr_[i];
  }
  const T& operator[](int i) const {
    return ptr_[i];
  }
  T* operator->() {
    return ptr_;
  }
  const T* operator->() const {
    return ptr_;
  }
};

#endif  // SRC_COMMON_X_H_
