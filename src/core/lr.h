// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// logistic regression
//

#ifndef SRC_CORE_LR_H_
#define SRC_CORE_LR_H_

#include "common/problem.h"

class LRModel {
 private:
  // Parameters are here for practical and convenience reasons.
  double eps_;  // stopping criteria
  double l1_c_;  // penalties on l1-norm of weights
  double l2_c_;  // penalties on l2-norm of weights
  // weights of all positive samples,
  // while weights of negative samples are all 1.0.
  double positive_weight_;

  // Parameters used in only FTRL mode(TrainFTRL).
  double ftrl_alpha_;
  double ftrl_beta_;

  // Real model stuffs here, they will be filled by TrainXXX functions.
  double bias_;  // no bias term if <= 0
  int columns_;  // number of features
  double* w_;  // weights

 public:
  double& eps() {
    return eps_;
  }
  double eps() const {
    return eps_;
  }
  double& l1_c() {
    return l1_c_;
  }
  double l1_c() const {
    return l1_c_;
  }
  double& l2_c() {
    return l2_c_;
  }
  double l2_c() const {
    return l2_c_;
  }
  double& positive_weight() {
    return positive_weight_;
  }
  double positive_weight() const {
    return positive_weight_;
  }
  double& ftrl_alpha() {
    return ftrl_alpha_;
  }
  double ftrl_alpha() const {
    return ftrl_alpha_;
  }
  double& ftrl_beta() {
    return ftrl_beta_;
  }
  double ftrl_beta() const {
    return ftrl_beta_;
  }
  double& bias() {
    return bias_;
  }
  double bias() const {
    return bias_;
  }
  int& columns() {
    return columns_;
  }
  int columns() const {
    return columns_;
  }
  double* w() {
    return w_;
  }
  const double* w() const {
    return w_;
  }

 private:
  static int PredictFeatureNodeProc(
    double bias,
    int with_label,
    int sort_x_by_index,
    void* arg,
    double y,
    int sample_max_column,
    FeatureNodeVector* x,
    int error_flag);

 public:
  LRModel();
  ~LRModel();
  void Clear();

  // L1: LBFGS + OWLQN
  // L2: LBFGS
  void TrainLBFGS(const Problem& problem);
  // TODO(yafei) TrainFTRL, UpdateFTRL
  void TrainFTRL(const Problem& problem);
  void UpdateFTRL(const FeatureNode* node);

  double Predict(const FeatureNode* node) const;
  void Predict(FILE* fin, FILE* fout, int with_label) const;

  void Load(FILE* fp);
  void Save(FILE* fp) const;
};

#endif  // SRC_CORE_LR_H_
