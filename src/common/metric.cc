// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// model performance evaluation metrics
//

#include "common/metric.h"
#include "common/x.h"

void Evaluate(const std::vector<double>& pred,
              const std::vector<double>& y,
              BinaryClassificationMetric* metric,
              double theshold) {
  int tp = 0, fp = 0, fn = 0, tn = 0;
  int size = (int)y.size();
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
    Debug("No positive predicted samples(tp + fp = 0).\n");
    precision = 0.0;
  } else {
    precision = tp / (double) (tp + fp);
  }

  if (tp + fn == 0) {
    Debug("No positive true samples(tp + fn = 0).\n");
    recall = 0.0;
  } else {
    recall = tp / (double) (tp + fn);
  }

  if (precision + recall == 0) {
    Debug("precision + recall = 0.\n");
    fscore = 0.0;
  } else {
    fscore = 2 * precision * recall / (precision + recall);
  }

  accuracy = (tp + tn) / (double)size;

  Debug("Precision = %lg (%d/%d).\n", precision, tp, tp + fp);
  Debug("Recall = %lg (%d/%d).\n", recall, tp, tp + fn);
  Debug("FScore = %lg.\n", fscore);
  Debug("Accuracy = %lg (%d/%d).\n", accuracy, tp + tn, (int)size);

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

  bool operator()(int i, int j) const {
    return pred_[i] > pred_[j];
  }
};

double EvaluateAUC(const std::vector<double>& pred,
                   const std::vector<double>& y) {
  int size = (int)y.size();
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
    Debug("No true positive or true nagetive samples.\n");
    auc = 0.0;
  } else {
    auc = auc / tp / fp;
  }

  Debug("AUC = %lg.\n\n", auc);
  return auc;
}
