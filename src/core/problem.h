// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// features, samples and problem set
//

#ifndef SRC_CORE_PROBLEM_H_
#define SRC_CORE_PROBLEM_H_

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "core/x.h"

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

typedef std::map<std::string, int> FeatureMap;
typedef FeatureMap::const_iterator FeatureMapCI;

typedef std::multimap<int, std::string> FeatureReverseMap;
typedef FeatureReverseMap::const_iterator FeatureReverseMapCI;
typedef std::pair <FeatureReverseMap::const_iterator,
        FeatureReverseMap::const_iterator> FeatureReverseMapCII;

void FeatureMapToFeatureReverseMap(
  const FeatureMap& feature_map,
  FeatureReverseMap* fr_map);

enum FeatureNodeErrorFlag {
  Success = 0,
  LineEmpty = 1,
  LabelError = 2,
  FeatureEmpty = 4,
};

typedef void (* FeatureNodeProc)(
  int with_label,
  int sort_x_by_index,
  void* callback_arg,
  double y,
  int sample_max_column,
  FeatureNodeVector* x,
  int error_flag);

typedef void (* FeatureNodeHashProc)(
  void* callback_arg,
  const std::string& name,
  int* hashed_index);

void ForeachFeatureNode(
  FILE* fp,
  int with_label,
  int sort_x_by_index,
  void* callback_arg,
  FeatureNodeProc callback);

void ForeachFeatureNode_Hash(
  FILE* fp,
  int with_label,
  int sort_x_by_index,
  void* callback_arg,
  FeatureNodeProc callback,
  FeatureNodeHashProc hasher);

class Problem {
 private:
  int columns_;  // number of features
  std::vector<double> y_;
  std::vector<int> x_index_;
  int own_x_space_;
  FeatureNodeVector* x_space_;

 private:
  static void LoadFileProc(
    int with_label,
    int sort_x_by_index,
    void* callback_arg,
    double y,
    int sample_max_column,
    FeatureNodeVector* x,
    int error_flag);

  static void LoadFileHashProc(
    void* callback_arg,
    const std::string& name,
    int* hashed_index);

 public:
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
  void LoadFile(FILE* fp);
  // format label name:value[ name:value]
  // feature names are hashed into [1, dimension].
  // "fr_map" is optional.
  void LoadHashFile(
    FILE* fp,
    int dimension,
    FeatureReverseMap* fr_map);

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

#endif  // SRC_CORE_PROBLEM_H_
