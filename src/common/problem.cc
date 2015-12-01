// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//

#include <algorithm>

#include "problem.h"
#include "line-reader.h"

static const char* kDelimiter = " \t|\n";

bool Problem::LoadText(FILE* fp) {
  Clear();

  LineReader line_reader;
  int64_t x_space_size = 0;
  int64_t max_column = 0, instance_max_column, i = 0, j = 0, k;
  int64_t need_sort;
  char* endptr;
  char* label;
  char* index;
  char* value;

  // Debug("1st turn.\n");
  while (line_reader.ReadLine(fp) != NULL) {
    // label
    label = strtok(line_reader.buf, kDelimiter);
    if (label == NULL) {
      // empty line
      continue;
    }

    // features
    for (;;) {
      index = strtok(NULL, kDelimiter);
      if (index == NULL) {
        break;
      }
      x_space_size++;
    }
    if (bias >= 0) {
      x_space_size++;
    }
    rows++;
  }
  Debug("rows=%lld\n", rows);
  Debug("x_space_size=%lld\n", x_space_size);

  // Debug("2nd turn.\n");
  rewind(fp);
  y = Malloc(double, rows);
  x = Malloc(FeatureNode*, rows);
  x_space = Malloc(FeatureNode, x_space_size + rows);
  while (line_reader.ReadLine(fp) != NULL) {
    label = strtok(line_reader.buf, kDelimiter);
    if (label == NULL) {
      // empty line
      continue;
    }

    instance_max_column = 0;
    k = 0;
    need_sort = 0;
    x[i] = &x_space[j];
    y[i] = strtod(label, &endptr);
    if (endptr == label || *endptr != '\0') {
      Error("line %lld, label error.\n", i + 1);
      Clear();
      return false;
    }

    for (;;) {
      index = strtok(NULL, kDelimiter);
      if (index == NULL) {
        break;
      }

      value = strchr(index, ':');
      if (value) {
        *value = '\0';
        value++;
        x_space[j].value = strtod(value, &endptr);
        if (endptr == value || *endptr != '\0') {
          Error("line %lld, feature value error \"%s\".\n", i + 1, value);
          Clear();
          return false;
        }
      } else {
        x_space[j].value = 1.0;
      }

      x_space[j].index = (int64_t)strtoll(index, &endptr, 10);
      if (endptr == index || *endptr != '\0') {
        Error("line %lld, feature index error \"%s\".\n", i + 1, index);
        Clear();
        return false;
      }
      if (x_space[j].index > instance_max_column) {
        instance_max_column = x_space[j].index;
      } else {
        need_sort = 1;
      }

      j++;
      k++;
    }
    if (need_sort) {
      std::sort(x[i], x[i] + k, FeatureNodeLess());
    }
    if (instance_max_column > max_column) {
      max_column = instance_max_column;
    }

    if (bias >= 0) {
      x_space[j++].value = bias;
    }

    // a sentinel
    x_space[j].index = -1;
    x_space[j++].value = -1;

    i++;
  }

  if (bias >= 0) {
    columns = max_column + 1;
    for (i = 1; i < rows; i++) {
      // assign bias term's index
      (x[i] - 2)->index = columns;
    }
    x_space[j - 2].index = columns;
  } else {
    columns = max_column;
  }

  // for (i = 0; i < x_space_size + rows; i++) {
  // Debug("x_space[%lld]=%lld:%g\n", i,
  // x_space[i].index, x_space[i].value);
  // }

  return true;
}

bool Problem::Load(FILE* fp) {
  return true;
}

bool Problem::Save(FILE* fp) const {
  return true;
}
