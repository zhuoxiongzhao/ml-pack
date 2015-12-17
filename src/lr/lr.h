// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// logistic regression
//

#ifndef SRC_CORE_LR_H_
#define SRC_CORE_LR_H_

#include "lr/problem.h"

class LRModel {
 private:
  // Parameters for both LBFGS and FTRL mode.
  double eps_;  // stopping criteria
  double l1_c_;  // penalties on l1-norm of weights
  double l2_c_;  // penalties on l2-norm of weights
  // weights of all positive samples,
  // while weights of negative samples are all 1.0.
  double positive_weight_;
  double bias_;  // no bias term if <= 0

  // Parameters used in only FTRL mode.
  double ftrl_alpha_;
  double ftrl_beta_;
  int ftrl_round_;

  // Real model stuffs here, they will be filled by TrainXXX functions.
  int columns_;  // number of features including the bias term
  double* w_;  // weights for both TrainLBFGS and TrainFTRL
  double* ftrl_zn_;  // context only for TrainFTRL

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
  double& bias() {
    return bias_;
  }
  double bias() const {
    return bias_;
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
  int& ftrl_round() {
    return ftrl_round_;
  }
  int ftrl_round() const {
    return ftrl_round_;
  }

  int columns() const {
    return columns_;
  }

 public:
  LRModel();
  ~LRModel();

  // Only clear model stuffs, won't reset parameters.
  void Clear();

  // L1: LBFGS + OWLQN
  // L2: LBFGS
  void TrainLBFGS(const Problem& problem);
  // L1 and L2: FTRL
  void TrainFTRL(const Problem& problem);
  void UpdateFTRL(double y, const FeatureNode* x);

  // mode 0: LBFGS
  // mode 1: FTRL
  void Train(const Problem& problem, int mode);

  double Predict(const FeatureNode* x) const;
  void PredictFile(FILE* fin, FILE* fout, int with_label) const;
  void PredictHashFile(FILE* fin,
                       FILE* fout,
                       int with_label,
                       int dimension) const;

  void Load(FILE* fp);
  void Save(FILE* fp, const FeatureReverseMap* fr_map = NULL) const;
};

#endif  // SRC_CORE_LR_H_
