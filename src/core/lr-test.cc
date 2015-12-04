// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//

#include "core/lr.h"
#include "common/metric.h"

int main() {
  Problem problem;
#if defined _WIN32
  FILE* fp = xfopen("../src/test-data/heart_scale", "r");
#else
  FILE* fp = xfopen("src/test-data/heart_scale", "r");
#endif
  problem.LoadText(fp, 1.0);
  fclose(fp);

  LRModel model;
  model.l1_c = 0.0;
  model.l2_c = 1.0;
  model.TrainLBFGS(problem);
  model.Save(stdout);

  model.l1_c = 1.0;
  model.l2_c = 0.0;
  model.TrainLBFGS(problem);
  model.Save(stdout);

  double* pred = Malloc(double, problem.rows);
  ScopedPtr<double*> guard(pred);
  for (int i = 0; i < problem.rows; i++) {
    pred[i] = model.Predict(problem.x[i]);
  }

  BinaryClassificationMetric metric;
  Evaluate(pred, problem.y, problem.rows, &metric);
  return 0;
}
