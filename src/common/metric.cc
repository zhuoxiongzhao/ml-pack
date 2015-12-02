// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// model performance evaluation metrics
//

#include <algorithm>

#include "common/metric.h"
#include "common/x.h"

void Evaluate(const DoubleVector& pred,
              const DoubleVector& y,
              BinaryClassificationMetric* metric,
              double theshold) {
  size_t size = pred.size();
  int tp = 0, fp = 0, fn = 0, tn = 0;
  double precision, recall, fscore, accuracy;

  for (size_t i = 0; i < size; i++) {
    if (pred[i] >= theshold) {
      if (y[i] == 1.0) {
        tp++;
      } else {
        fp++;
      }
    } else {
      if (y[i] == 1.0) {
        fn++;
      } else {
        tn++;
      }
    }
  }

  if (tp + fp == 0) {
    Log("No positive predicted samples(tp + fp = 0).\n");
    precision = 0;
  } else {
    precision = tp / (double) (tp + fp);
  }

  if (tp + fn == 0) {
    Log("No positive true samples(tp + fn = 0).\n");
    recall = 0;
  } else {
    recall = tp / (double) (tp + fn);
  }

  if (precision + recall == 0) {
    Log("precision + recall = 0.\n");
    fscore = 0;
  } else {
    fscore = 2 * precision * recall / (precision + recall);
  }

  accuracy = (tp + tn) / (double)size;

  Log("Precision = %g (%d/%d).\n", precision, tp, tp + fp);
  Log("Recall = %g (%d/%d).\n", recall, tp, tp + fn);
  Log("FScore = %g.\n", fscore);
  Log("Accuracy = %g (%d/%d).\n", accuracy, tp + tn, (int)size);

  metric->precision = precision;
  metric->recall = recall;
  metric->fscore = fscore;
  metric->accuracy = accuracy;
  metric->auc = EvaluateAUC(pred, y);
}

class IndicesCompare {
 private:
  const double* pred_;
 public:
  explicit IndicesCompare(const double* pred) : pred_(pred) {}

  bool operator()(size_t i, size_t j) const {
    return pred_[i] > pred_[j];
  }
};

double EvaluateAUC(const DoubleVector& pred,
                   const DoubleVector& y) {
  size_t size = pred.size();
  std::vector<size_t> indices(size);
  int tp = 0, fp = 0;
  double auc = 0;

  for (size_t i = 0; i < size; i++) {
    indices[i] = i;
  }

  std::sort(indices.begin(), indices.end(), IndicesCompare(&pred[0]));

  for (size_t i = 0; i < size; i++) {
    if (y[indices[i]] == 1) {
      tp++;
    } else if (y[indices[i]] == -1) {
      auc += tp;
      fp++;
    }
  }

  if (tp == 0 || fp == 0) {
    Log("No true positive or true nagetive samples.\n");
    auc = 0;
  } else {
    auc = auc / tp / fp;
  }

  Log("AUC = %g.\n", auc);
  return auc;
}
