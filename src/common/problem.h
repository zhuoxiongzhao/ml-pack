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
  int pad;
  double value;
};

struct FeatureNodeLess {
  bool operator()(const FeatureNode& a, const FeatureNode& b) const {
    return a.index < b.index;
  }
};

struct Problem {
  double bias;  // < 0 if no bias term
  int rows;
  int columns;
  int x_space_size;
  double* y;
  FeatureNode** x;
  FeatureNode* x_space;

  Problem() : bias(1.0), rows(0), columns(0), x_space_size(0),
    y(NULL), x(NULL), x_space(NULL) {}

  ~Problem() {
    Clear();
  }

  void Clear() {
    rows = 0;
    columns = 0;
    x_space_size = 0;
    free(y);
    y = NULL;
    free(x);
    x = NULL;
    free(x_space);
    x_space = NULL;
  }

  // LIBSVM format
  bool LoadText(FILE* fp);
  bool LoadBinary(FILE* fp);
  bool SaveBinary(FILE* fp) const;
};

#endif  // SRC_COMMON_PROBLEM_H_
