// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// tests for lr.cc
//

#include "core/lr.h"
#include "core/metric.h"

int main() {
  Problem problem;

  {
#if defined _WIN32
    ScopedFile fp("../src/test-data/heart_scale", ScopedFile::Read);
#else
    ScopedFile fp("test-data/heart_scale", ScopedFile::Read);
#endif
    problem.LoadText(fp, 1.0);
  }

  LRModel model;
  model.l1_c() = 0.0;
  model.l2_c() = 1.0;
  model.TrainLBFGS(problem);
  model.Save(stdout);
  model.Clear();

  model.l1_c() = 1.0;
  model.l2_c() = 0.0;
  model.TrainLBFGS(problem);
  model.Save(stdout);

  std::vector<double> pred;
  pred.resize(problem.rows());
  for (int i = 0; i < problem.rows(); i++) {
    pred[i] = model.Predict(problem.x(i));
  }

  BinaryClassificationMetric metric;
  Evaluate(pred, problem.y(), &metric);
  return 0;
}
