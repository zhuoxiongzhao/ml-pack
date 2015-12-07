// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//

#include "common/problem.h"
#include "common/line-reader.h"

void Problem::Clear() {
  bias = 1.0;
  rows = 0;
  columns = 0;
  x_space_size = 0;
  y.Free();
  x.Free();
  x_space.Free();
}

void Problem::LoadText(FILE* fp, double _bias) {
  LineReader line_reader;
  int max_column = 0, sample_max_column, i = 0, j = 0, k;
  char* endptr;
  char* label;
  char* index;
  char* value;

  Clear();
  bias = _bias;

  // Debug("1st turn.\n");
  while (line_reader.ReadLine(fp) != NULL) {
    // label
    label = strtok(line_reader.buf, DELIMITER);
    if (label == NULL) {
      // empty line
      continue;
    }

    // features
    for (;;) {
      index = strtok(NULL, DELIMITER);
      if (index == NULL) {
        break;
      }
      x_space_size++;
    }
    if (bias >= 0.0) {
      x_space_size++;
    }
    rows++;
  }
  Debug("rows=%d\n", rows);
  Debug("x_space_size=%d\n", x_space_size);

  // Debug("2nd turn.\n");
  rewind(fp);
  y.Malloc(rows);
  x.Malloc(rows);
  x_space.Malloc(x_space_size + rows);
  while (line_reader.ReadLine(fp) != NULL) {
    label = strtok(line_reader.buf, DELIMITER);
    if (label == NULL) {
      // empty line
      continue;
    }

    sample_max_column = 0;
    k = 0;
    x[i] = &x_space[j];
    y[i] = strtod(label, &endptr);
    if (*endptr != '\0') {
      Error("line %d, label error.\n", i + 1);
      exit(2);
    }

    for (;;) {
      index = strtok(NULL, DELIMITER);
      if (index == NULL) {
        break;
      }

      value = strrchr(index, ':');
      if (value) {
        if (value == index) {
          Error("line %d, feature index is empty.\n", i + 1);
          exit(3);
        }
        *value = '\0';
        value++;
        x_space[j].value = strtod(value, &endptr);
        if (*endptr != '\0') {
          Error("line %d, feature value error \"%s\".\n", i + 1, value);
          exit(4);
        }
      } else {
        x_space[j].value = 1.0;
      }

      x_space[j].index = (int)strtoll(index, &endptr, 10);
      if (*endptr != '\0') {
        Error("line %d, feature index error \"%s\".\n", i + 1, index);
        exit(5);
      }
      if (x_space[j].index > sample_max_column) {
        sample_max_column = x_space[j].index;
      }

      j++;
      k++;
    }

    if (sample_max_column > max_column) {
      max_column = sample_max_column;
    }

    if (bias >= 0.0) {
      x_space[j++].value = bias;
    }

    // a sentinel
    x_space[j++].index = -1;
    i++;
  }

  if (bias >= 0.0) {
    columns = max_column + 1;
    for (i = 1; i < rows; i++) {
      // assign bias term's index
      (x[i] - 2)->index = columns;
    }
    x_space[j - 2].index = columns;
  } else {
    columns = max_column;
  }
}

void Problem::LoadBinary(FILE* fp) {
  Clear();

  xfread(&bias, sizeof(bias), 1, fp);
  xfread(&rows, sizeof(rows), 1, fp);
  xfread(&columns, sizeof(columns), 1, fp);
  xfread(&x_space_size, sizeof(x_space_size), 1, fp);

  y.Malloc(rows);
  x.Malloc(rows);
  x_space.Malloc(x_space_size + rows);

  xfread(y, sizeof(y[0]), rows, fp);
  xfread(x_space, sizeof(x_space[0]), x_space_size + rows, fp);

  for (int i = 0, j = 0; i < rows;) {
    x[i] = &x_space[j];
    for (;;) {
      if (x_space[j].index != -1) {
        j++;
      } else {
        // sentinel
        j++;
        i++;
        break;
      }
    }
  }
}

void Problem::SaveBinary(FILE* fp) const {
  xfwrite(&bias, sizeof(bias), 1, fp);
  xfwrite(&rows, sizeof(rows), 1, fp);
  xfwrite(&columns, sizeof(columns), 1, fp);
  xfwrite(&x_space_size, sizeof(x_space_size), 1, fp);
  xfwrite(y, sizeof(y[0]), rows, fp);
  xfwrite(x_space, sizeof(x_space[0]), x_space_size + rows, fp);
}

void Problem::GenerateNFold(
  Problem* nfold_training,
  Problem* nfold_testing,
  int nfold) const {
  int piece = rows / nfold;
  int remains = rows % nfold;

  for (int i = 0; i < nfold; i++) {
    Problem* training, * testing;
    int k;
    int testing_rows = piece, training_rows;
    int testing_begin = i * piece;
    int testing_end = testing_begin + piece;
    if (i == nfold - 1) {
      testing_rows += remains;
      testing_end += remains;
    }
    training_rows = rows - testing_rows;

    training = nfold_training + i;
    training->bias = bias;
    training->rows = training_rows;
    training->columns = columns;
    training->y.Malloc(training_rows);
    training->x.Malloc(training_rows);

    testing = nfold_testing + i;
    testing->bias = bias;
    testing->rows = testing_rows;
    testing->columns = columns;
    testing->y.Malloc(testing_rows);
    testing->x.Malloc(testing_rows);

    for (int j = 0; j < testing_begin; j++) {
      training->y[j] = y[j];
      training->x[j] = x[j];
    }
    for (int j = testing_begin; j < testing_end; j++) {
      k = j - testing_begin;
      testing->y[k] = y[j];
      testing->x[k] = x[j];
    }
    for (int j = testing_end; j < rows; j++) {
      k = j - testing_end + testing_begin;
      training->y[k] = y[j];
      training->x[k] = x[j];
    }
  }
}

void Problem::GenerateTrainingTesting(
  Problem* training,
  Problem* testing,
  double testing_portion) const {
  int testing_rows = (int)(rows * testing_portion);
  int training_rows = rows - testing_rows;
  int k = 0;

  training->bias = bias;
  training->rows = training_rows;
  training->columns = columns;
  training->y.Malloc(training_rows);
  training->x.Malloc(training_rows);
  for (int j = 0; j < training_rows; j++, k++) {
    training->y[j] = y[k];
    training->x[j] = x[k];
  }

  testing->bias = bias;
  testing->rows = testing_rows;
  testing->columns = columns;
  testing->y.Malloc(testing_rows);
  testing->x.Malloc(testing_rows);
  for (int j = 0; j < testing_rows; j++, k++) {
    testing->y[j] = y[k];
    testing->x[j] = x[k];
  }
}
