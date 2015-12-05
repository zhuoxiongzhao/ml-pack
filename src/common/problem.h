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
  double bias;  // < 0 if no bias term
  int rows;  // number of samples
  int columns;  // number of features
  int x_space_size;

  ScopedPtr<double> y;
  ScopedPtr<FeatureNode*> x;
  ScopedPtr<FeatureNode> x_space;

  Problem() : bias(1.0), rows(0), columns(0), x_space_size(0) {}

  void Clear();
  // X format(fully compatible with LIBSVM format)
  // "_bias" < 0 if no bias term
  bool LoadText(FILE* fp, double _bias);
  void LoadBinary(FILE* fp);
  void SaveBinary(FILE* fp) const;
};

// TODO(yafei) make cross validation problem

#endif  // SRC_COMMON_PROBLEM_H_
