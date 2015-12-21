// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//

#include <algorithm>

#include "common/hash-entry.h"
#include "common/line-reader.h"
#include "lr/problem.h"

void FeatureMapToFeatureReverseMap(
  const FeatureMap& feature_map,
  FeatureReverseMap* fr_map) {
  fr_map->clear();
  FeatureMapCI it = feature_map.begin();
  FeatureMapCI last = feature_map.end();
  for (; it != last; ++it) {
    fr_map->insert(std::make_pair(it->second, it->first));
  }
}

void ForeachFeatureNode(
  FILE* fp,
  int with_label,
  int sort_x_by_index,
  void* callback_arg,
  FeatureNodeProc callback) {
  LineReader line_reader;
  int line_no = 0;
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
    line_no++;

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
    } else {
      FeatureNode sentinel;
      sentinel.index = -1;
      sentinel.value = 0.0f;
      x.push_back(sentinel);
    }

callback_label:
    callback(with_label, sort_x_by_index, callback_arg,
             y, sample_max_column, &x, error_flag);
  }
}

void ForeachFeatureNode_Hash(
  FILE* fp,
  int with_label,
  int sort_x_by_index,
  void* callback_arg,
  FeatureNodeProc callback,
  FeatureNodeHashProc hasher) {
  LineReader line_reader;
  int line_no = 0;
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

  feature_error_flag = 0;
  while (line_reader.ReadLine(fp) != NULL) {
    line_no++;

    error_flag = Success;
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

      if (feature_error_flag == 0) {
        hasher(callback_arg, index, &xj.index);
        x.push_back(xj);
      }
    }

    if (sort_x_by_index) {
      std::sort(x.begin(), x.end(), FeatureNodeLess());
    }
    if (x.empty()) {
      error_flag |= FeatureEmpty;
    } else {
      FeatureNode sentinel;
      sentinel.index = -1;
      sentinel.value = 0.0f;
      x.push_back(sentinel);
    }

callback_label:
    callback(with_label, sort_x_by_index, callback_arg,
             y, sample_max_column, &x, error_flag);
  }
}

Problem::Problem() {
  columns_ = 0;
  own_x_space_ = 0;
  x_space_ = NULL;
}

Problem::~Problem() {
  Clear();
}

void Problem::Clear() {
  columns_ = 0;
  y_.clear();
  x_index_.clear();
  if (own_x_space_) {
    delete x_space_;
    own_x_space_ = 0;
  }
  x_space_ = NULL;
}

struct LoadFileProcArg {
  Problem* problem;
  FeatureMap* feature_map;
};

void Problem::LoadFileProc(
  int with_label,
  int sort_x_by_index,
  void* callback_arg,
  double y,
  int sample_max_column,
  FeatureNodeVector* x,
  int error_flag) {
  if (error_flag) {
    return;
  }

  Problem* problem = ((LoadFileProcArg*)callback_arg)->problem;
  if (sample_max_column > problem->columns_) {
    problem->columns_ = sample_max_column;
  }

  problem->y_.push_back(y);
  problem->x_index_.push_back(problem->x_index_.back() + (int)x->size());
  problem->x_space_->insert(problem->x_space_->end(), x->begin(), x->end());
}

void Problem::LoadFileHashProc(
  void* callback_arg,
  const std::string& name,
  int* hashed_index) {
  Problem* problem = ((LoadFileProcArg*)callback_arg)->problem;
  FeatureMap* feature_map = ((LoadFileProcArg*)callback_arg)->feature_map;
  *hashed_index = (unsigned int)HashString(name) % problem->columns() + 1;
  if (feature_map) {
    (*feature_map)[name] = *hashed_index;
  }
}

void Problem::LoadFile(FILE* fp) {
  Clear();

  LoadFileProcArg callback_arg = {this, NULL};
  x_index_.push_back(0);
  own_x_space_ = 1;
  x_space_ = new FeatureNodeVector();
  ForeachFeatureNode(fp, 1, 1, &callback_arg, &LoadFileProc);
  x_index_.pop_back();

  Log("Loaded %d*%d text samples\n", rows(), columns());
}

void Problem::LoadHashFile(
  FILE* fp,
  int dimension,
  FeatureReverseMap* fr_map) {
  Clear();

  FeatureMap* feature_map = NULL;
  if (fr_map) {
    feature_map = new FeatureMap();
  }
  LoadFileProcArg callback_arg = {this, feature_map};
  columns_ = dimension;
  x_index_.push_back(0);
  own_x_space_ = 1;
  x_space_ = new FeatureNodeVector();
  ForeachFeatureNode_Hash(fp, 1, 1, &callback_arg,
                          &LoadFileProc, &LoadFileHashProc);
  x_index_.pop_back();

  if (fr_map) {
    FeatureMapToFeatureReverseMap(*feature_map, fr_map);
    delete feature_map;
  }

  Log("Loaded %d*%d hash text samples\n", rows(), columns());
}

void Problem::LoadBinary(FILE* fp) {
  Clear();

  int _rows;
  int x_space_size;
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
      x_index_.push_back(i + 1);
    }
  }
  x_index_.pop_back();
}

void Problem::SaveBinary(FILE* fp) const {
  int _rows = rows();
  int x_space_size = (int)x_space_->size();
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
    training->columns_ = columns_;
    training->y_.reserve(training_rows);
    training->x_index_.reserve(training_rows);
    training->own_x_space_ = 0;
    training->x_space_ = x_space_;

    testing = nfold_testing + i;
    testing->Clear();
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
