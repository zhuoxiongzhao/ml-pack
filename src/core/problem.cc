// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//

#include <algorithm>

#include "core/problem.h"
#include "core/line-reader.h"
#include "core/hash-entry.h"

void ForeachFeatureNode(
  FILE* fp,
  int with_label,
  int sort_x_by_index,
  void* arg,
  FeatureNodeProc callback) {
  LineReader line_reader;
  int line_no = 1;
  char* endptr;
  char* label;
  char* index;
  char* value;
  char* feature_begin;
  int error_flag;
  int feature_error_flag;
  double y;
  int sample_max_column;
  FeatureNodeVector x;
  FeatureNode xj;

  while (line_reader.ReadLine(fp) != NULL) {
    error_flag = Success;
    feature_error_flag = 0;
    sample_max_column = 0;
    y = 0.0;
    x.clear();

    // I.label
    if (with_label) {
      label = strtok(line_reader.buf, DELIMITER);
      if (label == NULL) {
        // empty line
        error_flag |= LineEmpty;
        goto callback_label;
      }
      feature_begin = NULL;
      y = strtod(label, &endptr);
      if (*endptr != '\0') {
        Error("line %d, label error.\n", line_no);
        error_flag |= LabelError;
      }
    } else {
      feature_begin = line_reader.buf;
    }

    // II.features
    for (;;) {
      index = strtok(feature_begin, DELIMITER);
      feature_begin = NULL;
      if (index == NULL) {
        break;
      }

      value = strrchr(index, ':');
      if (value) {
        if (value == index) {
          Error("line %d, feature index is empty.\n", line_no);
          feature_error_flag = 1;
        }
        *value = '\0';
        value++;
        xj.value = (float)strtod(value, &endptr);
        if (*endptr != '\0') {
          Error("line %d, feature value error \"%s\".\n", line_no, value);
          feature_error_flag = 1;
        }
      } else {
        xj.value = 1.0f;
      }

      xj.index = (int)strtoll(index, &endptr, 10);
      if (*endptr != '\0') {
        Error("line %d, feature index error \"%s\".\n", line_no, index);
        feature_error_flag = 1;
      }
      if (xj.index == 0) {
        Error("line %d, feature index must start from 1.\n", line_no);
        feature_error_flag = 1;
      }

      if (feature_error_flag == 0) {
        if (xj.index > sample_max_column) {
          sample_max_column = xj.index;
        }
        x.push_back(xj);
      }
    }

    if (sort_x_by_index) {
      std::sort(x.begin(), x.end(), FeatureNodeLess());
    }
    if (x.empty()) {
      error_flag |= FeatureEmpty;
    }

callback_label:
    callback(with_label, sort_x_by_index, arg,
             y, sample_max_column, &x, error_flag);
    line_no++;
  }
}

void ForeachFeatureNameNode(
  FILE* fp,
  int with_label,
  int sort_x_by_index,
  void* arg,
  FeatureNameNodeProc callback) {
  LineReader line_reader;
  int line_no = 1;
  char* endptr;
  char* label;
  char* index;
  char* value;
  char* feature_begin;
  int error_flag;
  int feature_error_flag;
  double y;
  int sample_max_column;
  FeatureNameNodeVector x;
  FeatureNameNode xj;

  while (line_reader.ReadLine(fp) != NULL) {
    error_flag = Success;
    feature_error_flag = 0;
    sample_max_column = 0;
    y = 0.0;
    x.clear();

    // I.label
    if (with_label) {
      label = strtok(line_reader.buf, DELIMITER);
      if (label == NULL) {
        // empty line
        error_flag |= LineEmpty;
        goto callback_label;
      }
      feature_begin = NULL;
      y = strtod(label, &endptr);
      if (*endptr != '\0') {
        Error("line %d, label error.\n", line_no);
        error_flag |= LabelError;
      }
    } else {
      feature_begin = line_reader.buf;
    }

    // II.features
    for (;;) {
      index = strtok(feature_begin, DELIMITER);
      feature_begin = NULL;
      if (index == NULL) {
        break;
      }

      value = strrchr(index, ':');
      if (value) {
        if (value == index) {
          Error("line %d, feature index is empty.\n", line_no);
          feature_error_flag = 1;
        }
        *value = '\0';
        value++;
        xj.value = (float)strtod(value, &endptr);
        if (*endptr != '\0') {
          Error("line %d, feature value error \"%s\".\n", line_no, value);
          feature_error_flag = 1;
        }
      } else {
        xj.value = 1.0f;
      }

      xj.name = index;

      if (feature_error_flag == 0) {
        x.push_back(xj);
      }
    }

    if (x.empty()) {
      error_flag |= FeatureEmpty;
    }

callback_label:
    callback(with_label, sort_x_by_index, arg,
             y, sample_max_column, &x, error_flag);
    line_no++;
  }
}

Problem::Problem() {
  bias_ = 1.0;
  columns_ = 0;
  own_x_space_ = 0;
  x_space_ = NULL;
}

Problem::~Problem() {
  Clear();
}

void Problem::Clear() {
  bias_ = 1.0;
  columns_ = 0;
  y_.clear();
  x_index_.clear();
  if (own_x_space_) {
    delete x_space_;
  }
  own_x_space_ = 0;
  x_space_ = NULL;
}

int Problem::LoadTextProc(
  int with_label,
  int sort_x_by_index,
  void* arg,
  double y,
  int sample_max_column,
  FeatureNodeVector* x,
  int error_flag) {
  if (error_flag) {
    return error_flag;
  }

  Problem* problem = (Problem*)arg;
  if (sample_max_column > problem->columns_) {
    problem->columns_ = sample_max_column;
  }

  problem->y_.push_back(y);
  problem->x_index_.push_back(problem->x_index_.back() + (int)x->size() + 1);
  problem->x_space_->insert(problem->x_space_->end(), x->begin(), x->end());

  FeatureNode bias_term;
  bias_term.index = -1;
  bias_term.value = (float)problem->bias();
  problem->x_space_->push_back(bias_term);
  return 0;
}

void Problem::LoadText(FILE* fp, double _bias) {
  Clear();

  if (_bias <= 0.0) {
    _bias = 0.0;
  }

  bias_ = _bias;
  x_index_.push_back(0);
  own_x_space_ = 1;
  x_space_ = new FeatureNodeVector();
  ForeachFeatureNode(fp, 1, 1, this, &LoadTextProc);
  x_index_.pop_back();
  if (bias_ > 0.0) {
    columns_++;
  }
  Log("Load %d*%d text samples\n", rows(), columns());
}

int Problem::LoadHashTextProc(
  int with_label,
  int sort_x_by_index,
  void* arg,
  double y,
  int sample_max_column,
  FeatureNameNodeVector* x,
  int error_flag) {
  if (error_flag) {
    return error_flag;
  }

  Problem* problem = (Problem*)arg;
  problem->y_.push_back(y);
  problem->x_index_.push_back(problem->x_index_.back() + (int)x->size() + 1);

  int size = (int)x->size();
  for (int i = 0; i < size; i++) {
    FeatureNode xi;
    const FeatureNameNode& name_node = (*x)[i];
    if (name_node.name.empty()) {
      xi.index = -1;
    } else {
      xi.index = (unsigned int)HashString(name_node.name)
                 % problem->columns() + 1;
    }
    xi.value = name_node.value;
    problem->x_space_->push_back(xi);
  }

  FeatureNode bias_term;
  bias_term.index = -1;
  bias_term.value = (float)problem->bias();
  problem->x_space_->push_back(bias_term);
  return 0;
}

void Problem::LoadHashText(FILE* fp, double _bias, int dimension) {
  Clear();

  if (_bias <= 0.0) {
    _bias = 0.0;
  }

  bias_ = _bias;
  columns_ = dimension;
  x_index_.push_back(0);
  own_x_space_ = 1;
  x_space_ = new FeatureNodeVector();
  ForeachFeatureNameNode(fp, 1, 1, this, &LoadHashTextProc);
  x_index_.pop_back();
  if (bias_ > 0.0) {
    columns_++;
  }
  Log("Load %d*%d hash text samples\n", rows(), columns());
}

void Problem::LoadBinary(FILE* fp) {
  Clear();

  int _rows;
  int x_space_size;
  xfread(&bias_, sizeof(bias_), 1, fp);
  xfread(&_rows, sizeof(_rows), 1, fp);
  xfread(&columns_, sizeof(columns_), 1, fp);
  xfread(&x_space_size, sizeof(x_space_size), 1, fp);

  y_.resize(_rows);
  x_index_.resize(_rows);
  x_space_ = new FeatureNodeVector();
  x_space_->resize(x_space_size);

  xfread(&y_[0], sizeof(y_[0]), _rows, fp);
  xfread(&(*x_space_)[0], sizeof((*x_space_)[0]), x_space_size, fp);

  x_index_.push_back(0);
  for (int i = 0; i < x_space_size; i++) {
    if ((*x_space_)[i].index == -1) {
      x_index_.push_back(i);
    }
  }
  x_index_.pop_back();
}

void Problem::SaveBinary(FILE* fp) const {
  int _rows = rows();
  int x_space_size = (int)x_space_->size();
  xfwrite(&bias_, sizeof(bias_), 1, fp);
  xfwrite(&_rows, sizeof(_rows), 1, fp);
  xfwrite(&columns_, sizeof(columns_), 1, fp);
  xfwrite(&x_space_size, sizeof(x_space_size), 1, fp);
  xfwrite(&y_[0], sizeof(y_[0]), _rows, fp);
  xfwrite(&(*x_space_)[0], sizeof((*x_space_)[0]), x_space_size, fp);
}

void Problem::GenerateNFold(
  Problem* nfold_training,
  Problem* nfold_testing,
  int nfold) const {
  int _rows = rows();
  int piece = _rows / nfold;
  int remains = _rows % nfold;

  for (int i = 0; i < nfold; i++) {
    Problem* training, * testing;
    int testing_rows = piece, training_rows;
    int testing_begin = i * piece;
    int testing_end = testing_begin + piece;
    if (i == nfold - 1) {
      testing_rows += remains;
      testing_end += remains;
    }
    training_rows = _rows - testing_rows;

    training = nfold_training + i;
    training->Clear();
    training->bias_ = bias_;
    training->columns_ = columns_;
    training->y_.reserve(training_rows);
    training->x_index_.reserve(training_rows);
    training->own_x_space_ = 0;
    training->x_space_ = x_space_;

    testing = nfold_testing + i;
    testing->Clear();
    testing->bias_ = bias_;
    testing->columns_ = columns_;
    testing->y_.reserve(testing_rows);
    testing->x_index_.reserve(testing_rows);
    testing->own_x_space_ = 0;
    testing->x_space_ = x_space_;

    for (int j = 0; j < testing_begin; j++) {
      training->y_.push_back(y_[j]);
      training->x_index_.push_back(x_index_[j]);
    }
    for (int j = testing_begin; j < testing_end; j++) {
      testing->y_.push_back(y_[j]);
      testing->x_index_.push_back(x_index_[j]);
    }
    for (int j = testing_end; j < _rows; j++) {
      training->y_.push_back(y_[j]);
      training->x_index_.push_back(x_index_[j]);
    }
  }
}

void Problem::GenerateTrainingTesting(
  Problem* training,
  Problem* testing,
  double testing_portion) const {
  int _rows = rows();
  int testing_rows = (int)(_rows * testing_portion);
  int training_rows = _rows - testing_rows;
  int k = 0;

  training->Clear();
  training->bias_ = bias_;
  training->columns_ = columns_;
  training->y_.reserve(training_rows);
  training->x_index_.reserve(training_rows);
  training->own_x_space_ = 0;
  training->x_space_ = x_space_;

  for (int j = 0; j < training_rows; j++, k++) {
    training->y_.push_back(y_[k]);
    training->x_index_.push_back(x_index_[k]);
  }

  testing->Clear();
  testing->bias_ = bias_;
  testing->columns_ = columns_;
  testing->y_.reserve(testing_rows);
  testing->x_index_.reserve(testing_rows);
  testing->own_x_space_ = 0;
  testing->x_space_ = x_space_;

  for (int j = 0; j < testing_rows; j++, k++) {
    testing->y_.push_back(y_[k]);
    testing->x_index_.push_back(x_index_[k]);
  }
}
