// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// features, samples and problem set
//

#ifndef SRC_COMMON_PROBLEM_H_
#define SRC_COMMON_PROBLEM_H_

#include "common/x.h"

struct FeatureNode {
  int index;
  double value;
};

struct FeatureNodeLess {
  bool operator()(const FeatureNode& a, const FeatureNode& b) const {
    return a.index < b.index;
  }
};

struct Problem {
  double bias;  // no bias term if < 0
  int rows;  // number of samples
  int columns;  // number of features
  int x_space_size;

  ScopedPtr<double> y;
  ScopedPtr<FeatureNode*> x;
  ScopedPtr<FeatureNode> x_space;

  Problem() : bias(1.0), rows(0), columns(0), x_space_size(0) {}

  void Clear();
  // X format(fully compatible with LIBSVM format)
  // "_bias" no bias term if < 0
  void LoadText(FILE* fp, double _bias);
  void LoadBinary(FILE* fp);
  void SaveBinary(FILE* fp) const;

  // generate n-fold cross validation problems,
  // the following fields will be filled:
  // bias, rows, columns, y, x
  void GenerateNFold(
    Problem* nfold_training,
    Problem* nfold_testing,
    int nfold) const;
  // generate training and testing problems,
  // the following fields will be filled:
  // bias, rows, columns, y, x
  void GenerateTrainingTesting(
    Problem* training,
    Problem* testing,
    double testing_portion) const;
};

#endif  // SRC_COMMON_PROBLEM_H_
