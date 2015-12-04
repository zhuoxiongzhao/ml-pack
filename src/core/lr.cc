// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//

#include <math.h>

#include "core/lr.h"
#include "blas/blas-decl.h"
#include "lbfgs/lbfgs.h"

LRModel::LRModel() : eps(1e-6), l1_c(0.0), l2_c(0.0),
  positive_weight(1.0),
  bias(1.0), columns(0), w(NULL) {}

LRModel::~LRModel() {
  Clear();
}

void LRModel::Clear() {
  eps = 1e-6;
  l1_c = 0.0;
  l2_c = 0.0;
  positive_weight = 1.0;
  bias = 1.0;
  columns = 0;
  if (w) {
    vecfree(w);
    w = NULL;
  }
}

void LRModel::Load(FILE* fp) {
  fscanf(fp, "eps=%lg\n", &eps);
  fscanf(fp, "l1_c=%lg\n", &l1_c);
  fscanf(fp, "l2_c=%lg\n", &l2_c);
  fscanf(fp, "positive_weight=%lg\n", &positive_weight);
  fscanf(fp, "bias=%lg\n", &bias);
  fscanf(fp, "columns=%d\n", &columns);
  fscanf(fp, "eps=%lg\n", &eps);

  fscanf(fp, "\nweights:\n");
  w = vecalloc(columns);
  for (int i = 0; i < columns; i++) {
    fscanf(fp, "%lg\n", &w[i]);
  }
}

void LRModel::Save(FILE* fp) const {
  fprintf(fp, "eps=%lg\n", eps);
  fprintf(fp, "l1_c=%lg\n", l1_c);
  fprintf(fp, "l2_c=%lg\n", l2_c);
  fprintf(fp, "positive_weight=%lg\n", positive_weight);
  fprintf(fp, "bias=%lg\n", bias);
  fprintf(fp, "columns=%d\n", columns);

  fprintf(fp, "\nweights:\n");
  for (int i = 0; i < columns; i++) {
    fprintf(fp, "%lg\n", w[i]);
  }
}

struct LossParameter {
  const Problem* problem;
  LRModel* model;
};

static inline double sigmoid(double x) {
  return 1.0 / (1.0 + exp(-x));
}

static double LRLoss(
  void* instance,
  const int _n,
  const double* w,
  double* g,
  const double step) {
  static const double ZERO = 0.0;

  LossParameter* parameter = (LossParameter*)instance;
  const Problem& problem = *parameter->problem;
  LRModel* model = parameter->model;

  register int i, rows = problem.rows;
  register double wx, h, hh;
  register double fx = 0.0, weighted_rows = 0.0;
  register FeatureNode* node;
  register double positive_weight = model->positive_weight;

  dcopy(problem.columns, &ZERO, 0, g, 1);

  for (i = 0; i < rows; i++) {
    // h = sigmoid(w^t * x)
    wx = 0.0;
    for (node = problem.x[i]; node->index != -1; node++) {
      wx += node->value * w[node->index - 1];
    }
    h = sigmoid(wx);

    // accumulate loss function
    if (problem.y[i] == 1.0) {
      weighted_rows += positive_weight;
      fx -= positive_weight * log(h);
      hh = positive_weight * (h - 1.0);
    } else {
      weighted_rows += 1.0;
      fx -= log(1 - h);
      hh = h;
    }

    // gradient
    for (node = problem.x[i]; node->index != -1; node++) {
      g[node->index - 1] += (hh * node->value);
    }
  }

  // deal with L2-regularization
  if (model->l2_c != 0.0) {
    // penalty the weight of bias term or not?
    // fx = fx + C||w||/2
    fx += (model->l2_c / 2.0 * ddot(problem.columns, w, 1, w, 1));
    // g = g + Cw
    daxpy(problem.columns, model->l2_c, w, 1, g, 1);
  }

  // scale according to total sample weights
  fx = fx / weighted_rows;
  dscal(problem.columns, 1.0 / weighted_rows, g, 1);
  return fx;
}

static int LRProgress(
  void* instance,
  int n,
  const double* x,
  const double* g,
  const double fx,
  const double xnorm,
  const double gnorm,
  const double step,
  int k,
  int n_evaluate
) {
  Log("Iteration %d: fx=%lg xnorm=%lg gnorm=%lg.\n",
      k, fx, xnorm, gnorm);
  return 0;
}

void LRModel::TrainLBFGS(const Problem& problem) {
  LossParameter loss_param = {&problem, this};
  lbfgs_parameter_t param;

  lbfgs_default_parameter(&param);
  param.epsilon = eps;
  if (l1_c != 0.0) {
    // NOTE
    param.orthantwise_c = l1_c / (double)problem.rows;
    param.orthantwise_start = 0;
    param.orthantwise_end = problem.columns;
    param.linesearch = LBFGS_LINESEARCH_BACKTRACKING_WOLFE;
  }

  bias = problem.bias;
  columns = problem.columns;
  w = vecalloc(columns);
  int ret = lbfgs(columns, w, 0, &LRLoss, &LRProgress, &loss_param, &param);
  Log("Optimization terminated with status code = %d.\n\n", ret);
}

double LRModel::Predict(const FeatureNode* node) const {
  double wx = 0.0;
  for (; node->index != -1; node++) {
    wx += w[node->index - 1] * node->value;
  }
  return sigmoid(wx);
}
