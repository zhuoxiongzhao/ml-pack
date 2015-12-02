// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//

#include <assert.h>
#include <math.h>

#include "core/lr.h"
#include "blas/blas-decl.h"
#include "lbfgs/lbfgs.h"

static inline double sigmoid(double x) {
  return 1.0 / (1.0 + exp(-x));
}

//
//static double lr_sparse_loss(
//  void* instance,
//  const int _n,
//  const double* w,
//  double* g,
//  const double step) {
//  SparseUserData* user_data = (SparseUserData*)instance;
//  const SparseXYSet& xy_set = user_data->xy_set;
//  const Param& lr_param = user_data->lr_param;
//  size_t m = xy_set.size(), n = (size_t)_n;
//  size_t i, j, s;
//  double wx, h, hh, xy_weight;
//  double fx = 0.0, weighted_m = 0.0;
//  const double zero = 0.0;
//
//  assert(xy_set.x_columns() == n - 1);
//  dcopy((int)n, &zero, 0, g, 1);
//
//  for (i = 0; i < m; i++) {
//    const SparseXY& xy = xy_set.get(i);
//    const IndexValueVector& ivs = xy.X();
//    xy_weight = xy.weight();
//
//    weighted_m += xy_weight;
//    wx = 0.0;
//    for (j = 0, s = ivs.size(); j < s; j++) {
//      const IndexValue& iv = ivs[j];
//      wx += w[iv.index] * iv.value;
//    }
//    wx += w[n - 1];
//    h = sigmoid(wx);
//
//    if (xy.is_positive()) {
//      fx -= xy_weight * log(h);
//      hh = xy_weight * (h - 1.0);
//    } else {
//      fx -= xy_weight * log(1 - h);
//      hh = xy_weight * h;
//    }
//
//    for (j = 0, s = ivs.size(); j < s; j++) {
//      const IndexValue& iv = ivs[j];
//      g[iv.index] += (hh * iv.value);
//    }
//    g[n - 1] += hh;
//  }
//
//  if (lr_param.regularization == 2 && lr_param.C != 0.0) {
//    // fx=fx + C||w||/2
//    fx += (lr_param.C / 2.0 * ddot((int)n - 1, w, 1, w, 1));
//    // g=g + Cw
//    daxpy((int)n - 1, lr_param.C, w, 1, g, 1);
//  }
//
//  fx = fx / weighted_m;
//  dscal((int)n, 1.0 / weighted_m, g, 1);
//  return fx;
//}
//
//
//static int lr_progress(
//  void* instance,
//  int n,
//  const double* x,
//  const double* g,
//  const double fx,
//  const double xnorm,
//  const double gnorm,
//  const double step,
//  int k,
//  int n_evaluate
//) {
//  printf("iteration %d\n", k);
//  return 0;
//}
//
//static void train(
//  const Param& _lr_param,
//  DoubleVector* weights,
//  bool dense,
//  size_t n,
//  size_t m,
//  void* instance) {
//  optimize_t opt;
//  lbfgs_evaluate_t eval = dense ? lr_dense_loss : lr_sparse_loss;
//
//  Param lr_param = _lr_param;
//  check_param(&lr_param);
//
//  lbfgs_parameter_t param;
//  lbfgs_default_parameter(&param);
//  param.epsilon = 1e-6;
//  if (lr_param.regularization == 1 && lr_param.C != 0.0) {
//    param.orthantwise_c = lr_param.C / (double)m;
//    param.orthantwise_start = 0;
//    param.orthantwise_end = (int)n;
//    param.linesearch = LBFGS_LINESEARCH_BACKTRACKING_WOLFE;
//  }
//
//  if (lr_param.optimization == "lbfgs") {
//    opt = lbfgs;
//  } else if (lr_param.optimization == "cg") {
//    opt = cg;
//  } else {
//    opt = gd;
//  }
//
//  double* w = vecalloc(n + 1);
//  int ret = opt((int)n + 1, w, 0, eval, lr_progress, instance, &param);
//  printf("optimization terminated with status code = %d\n\n", ret);
//  weights->assign(w, w + n + 1);
//  vecfree(w);
//
//  printf("feature weight vector:\n");
//  for (size_t j = 0, s = weights->size(); j < s; j++) {
//    printf("    %8.8lf\n", (*weights)[j]);
//  }
//  printf("\n");
//}
//
//void LR::train(const DenseXYSet& xy_set) {
//  DenseUserData user_data(xy_set, lr_param_);
//  ::train(lr_param_, &w_, true, xy_set.x_columns(), xy_set.size(), (void*)&user_data);
//}
//
//void LR::train(const SparseXYSet& xy_set) {
//  SparseUserData user_data(xy_set, lr_param_);
//  ::train(lr_param_, &w_, false, xy_set.x_columns(), xy_set.size(), (void*)&user_data);
//}
//
//template <class XYSet>
//static void performance(const LR& lr, const XYSet& xy_set, LRPerformance* perf) {
//  double predicted_y;
//  bool trained, real;
//  size_t total = 0, TP = 0;
//  DoubleVector ys, predicted_ys;
//
//  ys.reserve(xy_set.size());
//  predicted_ys.reserve(xy_set.size());
//
//  for (size_t i = 0, s = xy_set.size(); i < s; i++) {
//    const typename XYSet::XYType& xy = xy_set.get(i);
//    predicted_y = lr.predict(xy.X());
//    trained = (predicted_y > 0.5);
//    real = xy.is_positive();
//
//    if (real) {
//      ys.push_back(1.0);
//    } else {
//      ys.push_back(-1.0);
//    }
//    predicted_ys.push_back(predicted_y);
//
//    if (trained == real) {
//      TP++;
//    }
//    total++;
//  }
//  perf->accuracy = (double)TP / (double)total;
//  perf->total = total;
//  perf->true_positive = TP;
//  perf->auc = ::auc(ys, predicted_ys);
//}
//
//void LR::performance(const DenseXYSet& xy_set, LRPerformance* perf) const {
//  ::performance(*this, xy_set, perf);
//}
//
//void LR::performance(const SparseXYSet& xy_set, LRPerformance* perf) const {
//  ::performance(*this, xy_set, perf);
//}
//
//double LR::predict(const DoubleVector& X) const {
//  assert(X.size() + 1 == w_.size());
//  double wx = 0.0;
//  size_t i, s;
//  for (i = 0, s = X.size(); i < s; i++) {
//    wx += w_[i] * X[i];
//  }
//  wx += w_[i];
//  return sigmoid(wx);
//}
//
//double LR::predict(const IndexValueVector& X) const {
//  double wx = 0.0;
//  size_t i, s;
//  for (i = 0, s = X.size(); i < s; i++) {
//    const IndexValue& iv = X[i];
//    wx += w_[iv.index] * iv.value;
//  }
//  wx += w_.back();
//  return sigmoid(wx);
//}
//
//int LR::save_model(FILE* fp) const {
//  for (size_t j = 0, s = w_.size(); j < s; j++) {
//    fprintf(fp, "%18.18lf\n", w_[j]);
//  }
//  return 0;
//}
//
//void LR::load_model(FILE* fp) {
//  int ret;
//  double w;
//  w_.clear();
//  for (;;) {
//    ret = fscanf(fp, "%lf\n", &w);
//    if (ret != 1) {
//      break;
//    }
//    w_.push_back(w);
//  }
//}
