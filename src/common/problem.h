// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// features, samples and problem set
//

#ifndef SRC_COMMON_PROBLEM_H_
#define SRC_COMMON_PROBLEM_H_

#include "common/x.h"

struct FeatureNode {
  int index;  // index starts from 1
  float value;
};

struct FeatureNodeLess {
  bool operator()(const FeatureNode& a, const FeatureNode& b) const {
    return a.index < b.index;
  }
};

typedef std::vector<FeatureNode> FeatureNodeVector;

struct FeatureNameNode {
  std::string name;
  float value;
};

typedef std::vector<FeatureNameNode> FeatureNameNodeVector;

enum ErrorFlag {
  Success = 0,
  LineEmpty = 1,
  LabelError = 2,
  FeatureEmpty = 4,
};

typedef int (* FeatureNodeProc)(
  double bias,
  int with_label,
  int sort_x_by_index,
  void* arg,
  double y,
  int sample_max_column,
  FeatureNodeVector* x,
  int error_flag);

void ForeachFeatureNode(
  FILE* fp,
  double bias,
  int with_label,
  int sort_x_by_index,
  void* arg,
  FeatureNodeProc callback);

typedef int (* FeatureNameNodeProc)(
  double bias,
  int with_label,
  int sort_x_by_index,
  void* arg,
  double y,
  int sample_max_column,
  FeatureNameNodeVector* x,
  int error_flag);

void ForeachFeatureNameNode(
  FILE* fp,
  double bias,
  int with_label,
  int sort_x_by_index,
  void* arg,
  FeatureNameNodeProc callback);

class Problem {
 private:
  double bias_;  // no bias term if <= 0
  int columns_;  // number of features including the bias term
  std::vector<double> y_;
  std::vector<int> x_index_;
  int own_x_space_;
  FeatureNodeVector* x_space_;

 private:
  static int LoadTextProc(
    double bias,
    int with_label,
    int sort_x_by_index,
    void* arg,
    double y,
    int sample_max_column,
    FeatureNodeVector* x,
    int error_flag);

  static int LoadHashTextProc(
    double bias,
    int with_label,
    int sort_x_by_index,
    void* arg,
    double y,
    int sample_max_column,
    FeatureNameNodeVector* x,
    int error_flag);

 public:
  double bias() const {
    return bias_;
  }

  int columns() const {
    return columns_;
  }

  int rows() const {
    return (int) y_.size();
  }

  const std::vector<double>& y() const {
    return y_;
  }

  double y(int i) const {
    return y_[i];
  }

  const FeatureNode* x(int i) const {
    return &((*x_space_)[0]) + x_index_[i];
  }

 public:
  Problem();
  ~Problem();
  void Clear();

  // format: label index:value[ index:value]
  // indices start from 1.
  void LoadText(FILE* fp, double _bias);
  // format label name:value[ name:value]
  // names are hashed into [1, dimension].
  void LoadHashText(FILE* fp, double _bias, int dimension);

  void LoadBinary(FILE* fp);
  void SaveBinary(FILE* fp) const;

  // generate n-fold cross validation problems
  void GenerateNFold(
    Problem* nfold_training,
    Problem* nfold_testing,
    int nfold) const;
  // generate training and testing problems
  void GenerateTrainingTesting(
    Problem* training,
    Problem* testing,
    double testing_portion) const;
};

#endif  // SRC_COMMON_PROBLEM_H_
