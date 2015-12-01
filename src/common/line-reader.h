// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// line reader
//

#ifndef SRC_COMMON_LINE_READER_H_
#define SRC_COMMON_LINE_READER_H_

#include "x.h"

struct LineReader {
  char* buf;
  size_t len;

  LineReader() : buf(NULL), len(4096) {
    buf = Malloc(char, 4096);
  }

  ~LineReader() {
    free(buf);
  }

  char* ReadLine(FILE* fp) {
    size_t new_len;
    if (fgets(buf, len, fp) == NULL) {
      return NULL;
    }

    while (strrchr(buf, '\n') == NULL) {
      len *= 2;
      buf = (char*)xrealloc(buf, len);
      new_len = strlen(buf);
      if (fgets(buf + new_len, len - new_len, fp) == NULL) {
        break;
      }
    }
    return buf;
  }
};

#endif  // SRC_COMMON_LINE_READER_H_
