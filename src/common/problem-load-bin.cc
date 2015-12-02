// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// load binary sample files
//

#include "common/problem.h"

int main(int argc, char** argv) {
  if (argc == 1) {
    fprintf(stderr, "%s SAMPLE_FILE1 [SAMPLE_FILE2] ...\n", argv[0]);
    return 1;
  }

  for (int i = 1; i < argc; i++) {
    Problem problem;
    FILE* fp = yfopen(argv[i], "rb");
    if (fp == NULL) {
      continue;
    }
    ScopedFile guard(fp);
    Log("Reading \"%s\"...\n", argv[i]);
    if (problem.LoadBinary(fp)) {
      Log("Done.\n");
    } else {
      Error("Failed.\n");
    }
  }
  return 0;
}
