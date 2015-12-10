// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// model performance evaluation metrics
//

#ifndef SRC_COMMON_METRIC_H_
#define SRC_COMMON_METRIC_H_

#include <vector>

struct BinaryClassificationMetric {
  double precision;
  double recall;
  double fscore;
  double accuracy;
  double auc;
};

void Evaluate(const std::vector<double>& pred,
              const std::vector<double>& y,
              BinaryClassificationMetric* metric,
              double theshold = 0.5);
double EvaluateAUC(const std::vector<double>& pred,
                   const std::vector<double>& y);

#endif  // SRC_COMMON_METRIC_H_
