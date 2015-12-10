// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// line reader
//

#ifndef SRC_CORE_LINE_READER_H_
#define SRC_CORE_LINE_READER_H_

#include "core/x.h"

struct LineReader {
 private:
  size_t len_;

 public:
  char* buf;

  LineReader() : len_(4096) {
    buf = _Malloc(char, len_);
  }

  ~LineReader() {
    free(buf);
  }

  char* ReadLine(FILE* fp) {
    size_t new_len;
    if (fgets(buf, len_, fp) == NULL) {
      return NULL;
    }

    while (strrchr(buf, '\n') == NULL) {
      len_ *= 2;
      buf = _Realloc(buf, char, len_);
      new_len = strlen(buf);
      if (fgets(buf + new_len, len_ - new_len, fp) == NULL) {
        break;
      }
    }
    return buf;
  }
};

#endif  // SRC_CORE_LINE_READER_H_
