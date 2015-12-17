// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// lr tests
//

#include "lr/lr.h"

int main() {
  Problem problem;

  {
#if defined _WIN32
    ScopedFile fp("../src/lr-test-data/heart_scale", ScopedFile::Read);
#else
    ScopedFile fp("lr-test-data/heart_scale", ScopedFile::Read);
#endif
    problem.LoadFile(fp);
  }

  LRModel model;
  model.l1_c() = 0.0;
  model.l2_c() = 1.0;
  model.bias() = 1.0;
  model.TrainLBFGS(problem);
  model.Save(stdout);

  model.l1_c() = 1.0;
  model.l2_c() = 0.0;
  model.bias() = 1.0;
  model.TrainLBFGS(problem);
  model.Save(stdout);

  model.l1_c() = 2.0;
  model.l2_c() = 1.0;
  model.ftrl_alpha() = 0.01;
  model.ftrl_beta() = 0.1;
  model.TrainFTRL(problem);
  model.TrainFTRL(problem);
  model.TrainFTRL(problem);
  model.Save(stdout);

  return 0;
}
