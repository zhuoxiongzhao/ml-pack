// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// line reader
//

#ifndef SRC_COMMON_LINE_READER_H_
#define SRC_COMMON_LINE_READER_H_

#include "common/x.h"

struct LineReader {
 private:
  size_t len_;
 public:
  ScopedPtr<char> buf;

  LineReader() : len_(4096) {
    buf.Malloc(len_);
  }

  char* ReadLine(FILE* fp) {
    size_t new_len;
    if (fgets(buf, len_, fp) == NULL) {
      return NULL;
    }

    // TODO(yafei) strrchr
    while (strrchr(buf, '\n') == NULL) {
      len_ *= 2;
      buf.Realloc(len_);
      new_len = strlen(buf);
      if (fgets(buf + new_len, len_ - new_len, fp) == NULL) {
        break;
      }
    }
    return buf;
  }
};

#endif  // SRC_COMMON_LINE_READER_H_
