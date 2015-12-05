// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// model performance evaluation metrics
//

#include <algorithm>
#include <vector>

#include "common/metric.h"
#include "common/x.h"

void Evaluate(const double* pred,
              const double* y,
              int size,
              BinaryClassificationMetric* metric,
              double theshold) {
  int tp = 0, fp = 0, fn = 0, tn = 0;
  double precision, recall, fscore, accuracy;

  for (int i = 0; i < size; i++) {
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
    precision = 0.0;
  } else {
    precision = tp / (double) (tp + fp);
  }

  if (tp + fn == 0) {
    Log("No positive true samples(tp + fn = 0).\n");
    recall = 0.0;
  } else {
    recall = tp / (double) (tp + fn);
  }

  if (precision + recall == 0) {
    Log("precision + recall = 0.\n");
    fscore = 0.0;
  } else {
    fscore = 2 * precision * recall / (precision + recall);
  }

  accuracy = (tp + tn) / (double)size;

  Log("Precision = %lg (%d/%d).\n", precision, tp, tp + fp);
  Log("Recall = %lg (%d/%d).\n", recall, tp, tp + fn);
  Log("FScore = %lg.\n", fscore);
  Log("Accuracy = %lg (%d/%d).\n", accuracy, tp + tn, (int)size);

  metric->precision = precision;
  metric->recall = recall;
  metric->fscore = fscore;
  metric->accuracy = accuracy;
  metric->auc = EvaluateAUC(pred, y, size);
}

class IndicesCompare {
 private:
  const double* pred_;
 public:
  explicit IndicesCompare(const double* pred) : pred_(pred) {}

  bool operator()(int i, int j) const {
    return pred_[i] > pred_[j];
  }
};

double EvaluateAUC(const double* pred,
                   const double* y,
                   int size) {
  std::vector<int> indices(size);
  int tp = 0, fp = 0;
  double auc = 0.0;

  for (int i = 0; i < size; i++) {
    indices[i] = i;
  }

  std::sort(indices.begin(), indices.end(), IndicesCompare(&pred[0]));

  for (int i = 0; i < size; i++) {
    if (y[indices[i]] == 1) {
      tp++;
    } else if (y[indices[i]] == -1) {
      auc += tp;
      fp++;
    }
  }

  if (tp == 0 || fp == 0) {
    Log("No true positive or true nagetive samples.\n");
    auc = 0.0;
  } else {
    auc = auc / tp / fp;
  }

  Log("AUC = %lg.\n", auc);
  return auc;
}