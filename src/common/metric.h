// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// model performance evaluation metrics
//

#ifndef SRC_COMMON_METRIC_H_
#define SRC_COMMON_METRIC_H_

struct BinaryClassificationMetric {
  double precision;
  double recall;
  double fscore;
  double accuracy;
  double loglikelyhood;  // TODO(yafei)
  double auc;
};

void Evaluate(const double* pred,
              const double* y,
              int size,
              BinaryClassificationMetric* metric,
              double theshold = 0.5);
double EvaluateAUC(const double* pred,
                   const double* y,
                   int size);

#endif  // SRC_COMMON_METRIC_H_
