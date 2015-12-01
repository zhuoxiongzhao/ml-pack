// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// load text problem files
//

#include "problem.h"

int main(int argc, char** argv) {
  if (argc == 1) {
    Log("%s file1 [file2] [file3] [...]\n", argv[0]);
    return 1;
  }

  for (int i = 1; i < argc; i++) {
    Problem problem;
    FILE* fp = yfopen(argv[i], "r");
    if (fp == NULL) {
      continue;
    }
    ScopedFile guard(fp);
    Log("Loading \"%s\"...\n", argv[i]);
    if (problem.LoadText(fp)) {
      Log("OK.\n");
    } else {
      Error("Failed.\n");
    }
  }
  return 0;
}
