// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// load binary sample files
//

#include "lr/problem.h"

int main(int argc, char** argv) {
  if (argc == 1) {
    fprintf(stderr, "%s SAMPLE_FILE1 [SAMPLE_FILE2] ...\n", argv[0]);
    return 1;
  }

  for (int i = 1; i < argc; i++) {
    Problem problem;
    ScopedFile fp(argv[i], ScopedFile::ReadBinary);
    if (fp == NULL) {
      continue;
    }
    Log("Reading \"%s\"...\n", argv[i]);
    problem.LoadBinary(fp);
    Log("Done.\n\n");
  }
  return 0;
}
