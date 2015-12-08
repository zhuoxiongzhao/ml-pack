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
      ScopedFile fin(argv[i], ScopedFile::Read);
      Log("Reading \"%s\"...\n", argv[i]);
      problem.LoadX(fin, 1.0);
      Log("Done.\n\n");
    }

    {
      std::string filename = argv[i];
      filename += ".bin";
      ScopedFile fout(filename.c_str(), ScopedFile::WriteBinary);
      Log("Writing \"%s\"...\n", filename.c_str());
      problem.SaveBinary(fout);
    }
  }
  return 0;
}
