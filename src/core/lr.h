// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// logistic regression
//

#ifndef SRC_CORE_LR_H_
#define SRC_CORE_LR_H_

#include "common/problem.h"

struct LRModel {
  // Parameters are here for practical and convenience reasons.
  double eps;  // stopping criteria
  double l1_c;  // penalties on l1-norm of weights
  double l2_c;  // penalties on l2-norm of weights
  // weights of all positive samples,
  // while weights of negative samples are all 1.0.
  double positive_weight;

  // Real model stuffs here, they will be filled by TrainXXX functions.
  double bias;  // < 0 if no bias term
  int columns;  // number of features
  double* w;  // weights

  LRModel();
  ~LRModel();
  void Clear();

  // L1: LBFGS + OWLQN
  // L2: LBFGS
  void TrainLBFGS(const Problem& problem);
  // TODO(yafei) TrainFTRL, UpdateFTRL
  void TrainLFTRL(const Problem& problem);
  void UpdateFTRL(const FeatureNode* node);

  double Predict(const FeatureNode* node) const;

  void Load(FILE* fp);
  void Save(FILE* fp) const;
};

#endif  // SRC_CORE_LR_H_
