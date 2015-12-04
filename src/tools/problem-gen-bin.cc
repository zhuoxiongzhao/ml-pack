// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// load text sample files and convert them into binary format
//

#include <string>

#include "common/problem.h"

int main(int argc, char** argv) {
  if (argc == 1) {
    fprintf(stderr, "%s SAMPLE_FILE1 [SAMPLE_FILE2] ...\n", argv[0]);
    return 1;
  }

  for (int i = 1; i < argc; i++) {
    Problem problem;

    {
      FILE* fin = xfopen(argv[i], "r");
      ScopedFile guard(fin);
      Log("Reading \"%s\"...\n", argv[i]);
      if (!problem.LoadText(fin, 1.0)) {
        continue;
        Error("Failed.\n");
      }
      Log("Done.\n");
    }

    {
      std::string filename = argv[i];
      filename += ".bin";
      FILE* fout = xfopen(filename.c_str(), "wb");
      ScopedFile guard(fout);
      Log("Writing \"%s\"...\n", argv[i]);
      problem.SaveBinary(fout);
    }
  }
  return 0;
}
