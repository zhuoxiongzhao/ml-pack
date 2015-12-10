// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//

#include "core/lr.h"
#include "core/hash-entry.h"
#include "core/line-reader.h"
#include "common/blas-decl.h"
#include "common/lbfgs.h"

LRModel::LRModel() {
  eps_ = 1e-6;
  l1_c_ = 0.0;
  l2_c_ = 0.0;
  positive_weight_ = 1.0;
  ftrl_alpha_ = 0.0;
  ftrl_beta_ = 0.0;
  bias_ = 1.0;
  columns_ = 0;
  w_ = NULL;
  ftrl_zn_ = NULL;
}

LRModel::~LRModel() {
  Clear();
}

void LRModel::Clear() {
  eps_ = 1e-6;
  l1_c_ = 0.0;
  l2_c_ = 0.0;
  positive_weight_ = 1.0;
  ftrl_alpha_ = 0.0;
  ftrl_beta_ = 0.0;
  bias_ = 1.0;
  columns_ = 0;
  if (w_) {
    vecfree(w_);
    w_ = NULL;
  }
  if (ftrl_zn_) {
    vecfree(ftrl_zn_);
    ftrl_zn_ = NULL;
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

  register int i;
  register int rows = problem.rows();
  int columns = problem.columns();
  double bias = problem.bias();
  register double wx, h, hh;
  register double fx = 0.0, weighted_rows = 0.0;
  register const FeatureNode* xi;
  register double positive_weight = model->positive_weight();

  dcopy(columns, &ZERO, 0, g, 1);

  for (i = 0; i < rows; i++) {
    // h = sigmoid(w^t * x)
    wx = 0.0;
    for (xi = problem.x(i); xi->index != -1; xi++) {
      wx += xi->value * w[xi->index - 1];
    }
    if (bias > 0.0) {
      wx += xi->value * w[columns - 1];
    }
    h = sigmoid(wx);

    // accumulate loss function
    if (problem.y(i) == 1.0f) {
      weighted_rows += positive_weight;
      fx -= positive_weight * log(h);
      hh = positive_weight * (h - 1.0);
    } else {
      weighted_rows += 1.0;
      fx -= log(1 - h);
      hh = h;
    }

    // gradient
    for (xi = problem.x(i); xi->index != -1; xi++) {
      g[xi->index - 1] += (hh * xi->value);
    }
    if (bias > 0.0) {
      g[columns - 1] += (hh * xi->value);
    }
  }

  // deal with L2-regularization
  if (model->l2_c() != 0.0) {
    // penalty the weight of bias term or not?
    // fx = fx + C||w||/2
    fx += (model->l2_c() / 2.0 * ddot(columns, w, 1, w, 1));
    // g = g + Cw
    daxpy(columns, model->l2_c(), w, 1, g, 1);
  }

  // scale according to total sample weights
  fx = fx / weighted_rows;
  dscal(columns, 1.0 / weighted_rows, g, 1);
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
  // Log("Iteration %d: fx=%lg xnorm=%lg gnorm=%lg.\n",
  //     k, fx, xnorm, gnorm);
  return 0;
}

void LRModel::TrainLBFGS(const Problem& problem) {
  LossParameter loss_param = {&problem, this};
  lbfgs_parameter_t param;
  lbfgs_default_parameter(&param);
  param.epsilon = eps_;
  if (l1_c_ != 0.0) {
    // NOTE
    param.orthantwise_c = l1_c_ / (double)problem.rows();
    param.orthantwise_start = 0;
    param.orthantwise_end = problem.columns();
    param.linesearch = LBFGS_LINESEARCH_BACKTRACKING_WOLFE;
  }

  bias_ = problem.bias();
  columns_ = problem.columns();
  w_ = vecalloc(columns_);
  int ret = lbfgs(columns_, w_, 0, &LRLoss, &LRProgress, &loss_param, &param);
  Log("Optimization terminated with status code = %d.\n\n", ret);
}

void LRModel::TrainFTRL(const Problem& problem) {
  bias_ = problem.bias();
  columns_ = problem.columns();
  w_ = vecalloc(columns_);
  ftrl_zn_ = vecalloc(columns_ * 2);
  // ftrl_zn_[2i + 0] is z[i]
  // ftrl_zn_[2i + 1] is n[i]

  int rows = problem.rows();
  for (int i = 0; i < rows; i++) {
    double y = problem.y(i);
    const FeatureNode* x = problem.x(i);
    UpdateFTRL(y, x);
  }
}

void LRModel::UpdateFTRL(double y, const FeatureNode* x) {
  double up, * n, * z;
  double g, n2, sqrt_n2, sigma;
  double flag, abs_z, step;
  int index;

  // up = -y/(1+exp(ywx))
  if (y == 1.0) {
    up = -positive_weight_ * sigmoid(-WX(x));
  } else {
    up = sigmoid(WX(x));
  }

  for (; x->index != -1; x++) {
    index = x->index - 1;
    n = ftrl_zn_ + 2 * index;
    z = n + 1;

    // 1. update n & z
    // g_i = -y/(1+exp(ywx)) * x_i = up * x_i
    g = up * x->value;
    n2 = *n + g * g;
    sqrt_n2 = sqrt(n2);
    // sigma = (sqrt(n+g^2) - sqrt(n)) / alpha
    sigma = (sqrt_n2 - sqrt(*n)) / ftrl_alpha_;
    // z = z + g - sigma*w
    *z += g - sigma * w_[index];
    // n = n + g^2
    *n = n2;

    // 2. update w
    flag = *z < 0.0 ? -1.0 : 1.0;
    abs_z = *z * flag;
    if (abs_z <= l1_c_) {
      w_[index] = 0.0;
    } else {
      step = 1.0 / (l2_c_ + (ftrl_beta_ + sqrt_n2) / ftrl_alpha_);
      w_[index] = flag * step * (l1_c_ - abs_z);
    }
  }

  if (bias_ > 0.0) {
    index = columns_ - 1;
    n = ftrl_zn_ + 2 * index;
    z = n + 1;

    // 1. update n & z
    // g_i = -y/(1+exp(ywx)) * x_i = up * x_i
    g = up * bias_;
    n2 = *n + g * g;
    sqrt_n2 = sqrt(n2);
    // sigma = (sqrt(n+g^2) - sqrt(n)) / alpha
    sigma = (sqrt_n2 - sqrt(*n)) / ftrl_alpha_;
    // z = z + g - sigma*w
    *z += g - sigma * w_[index];
    // n = n + g^2
    *n = n2;

    // 2. update w
    flag = *z < 0.0 ? -1.0 : 1.0;
    abs_z = *z * flag;
    if (abs_z <= l1_c_) {
      w_[index] = 0.0;
    } else {
      step = 1.0 / (l2_c_ + (ftrl_beta_ + sqrt_n2) / ftrl_alpha_);
      w_[index] = flag * step * (l1_c_ - abs_z);
    }
  }
}

double LRModel::WX(const FeatureNode* x) const {
  double wx = 0.0;
  for (; x->index != -1; x++) {
    wx += w_[x->index - 1] * x->value;
  }
  if (bias_ > 0.0) {
    wx += x->value * w_[columns_ - 1];
  }
  return wx;
}

double LRModel::Predict(const FeatureNode* x) const {
  double wx = 0.0;
  for (; x->index != -1; x++) {
    wx += w_[x->index - 1] * x->value;
  }
  if (bias_ > 0.0) {
    wx += x->value * w_[columns_ - 1];
  }
  return sigmoid(wx);
}

struct LRModelFout {
  const LRModel* model;
  FILE* fout;
};

int LRModel::PredictTextProc(
  int with_label,
  int sort_x_by_index,
  void* arg,
  double y,
  int sample_max_column,
  FeatureNodeVector* x,
  int error_flag) {
  const LRModel* model = ((LRModelFout*)arg)->model;
  FILE* fout = ((LRModelFout*)arg)->fout;

  if (error_flag == Success) {
    FeatureNode bias_term;
    bias_term.index = -1;
    bias_term.value = (float)model->bias();
    x->push_back(bias_term);
    fprintf(fout, "%lg", model->Predict(&(*x)[0]));
  }

  fprintf(fout, "\n");
  return 0;
}

void LRModel::PredictText(FILE* fin, FILE* fout, int with_label) const {
  LRModelFout arg;
  arg.model = this;
  arg.fout = fout;
  ::ForeachFeatureNode(fin, with_label, 0, &arg, &PredictTextProc);
}

int LRModel::PredictHashTextProc(
  int with_label,
  int sort_x_by_index,
  void* arg,
  double y,
  int sample_max_column,
  FeatureNameNodeVector* x,
  int error_flag) {
  const LRModel* model = ((LRModelFout*)arg)->model;
  FILE* fout = ((LRModelFout*)arg)->fout;

  if (error_flag == Success) {
    FeatureNodeVector x2;
    int size = (int)x->size();
    for (int i = 0; i < size; i++) {
      FeatureNode xi;
      const FeatureNameNode& name_node = (*x)[i];
      if (name_node.name.empty()) {
        xi.index = -1;
      } else {
        if (model->bias() > 0.0) {
          xi.index = (unsigned int)HashString(name_node.name)
                     % (model->columns() - 1) + 1;
        } else {
          xi.index = (unsigned int)HashString(name_node.name)
                     % model->columns() + 1;
        }
      }
      xi.value = name_node.value;
      x2.push_back(xi);
    }

    FeatureNode bias_term;
    bias_term.index = -1;
    bias_term.value = (float)model->bias();
    x2.push_back(bias_term);

    fprintf(fout, "%lg", model->Predict(&x2[0]));
  }

  fprintf(fout, "\n");
  return 0;
}

void LRModel::PredictHashText(FILE* fin, FILE* fout,
                              int with_label, int dimension) const {
  LRModelFout arg;
  arg.model = this;
  arg.fout = fout;
  ::ForeachFeatureNameNode(fin, with_label, 0, &arg, &PredictHashTextProc);
}

void LRModel::Load(FILE* fp) {
  fscanf(fp, "eps=%lg\n", &eps_);
  fscanf(fp, "l1_c=%lg\n", &l1_c_);
  fscanf(fp, "l2_c=%lg\n", &l2_c_);
  fscanf(fp, "positive_weight=%lg\n", &positive_weight_);
  fscanf(fp, "ftrl_alpha=%lg\n", &ftrl_alpha_);
  fscanf(fp, "ftrl_beta=%lg\n", &ftrl_beta_);
  fscanf(fp, "bias=%lg\n", &bias_);
  fscanf(fp, "columns=%d\n", &columns_);

  fscanf(fp, "\nweights:\n");
  w_ = vecalloc(columns_);
  for (int i = 0; i < columns_; i++) {
    fscanf(fp, "%lg\n", &w_[i]);
  }
}

void LRModel::Save(FILE* fp) const {
  fprintf(fp, "eps=%lg\n", eps_);
  fprintf(fp, "l1_c=%lg\n", l1_c_);
  fprintf(fp, "l2_c=%lg\n", l2_c_);
  fprintf(fp, "positive_weight=%lg\n", positive_weight_);
  fprintf(fp, "ftrl_alpha=%lg\n", ftrl_alpha_);
  fprintf(fp, "ftrl_beta=%lg\n", ftrl_beta_);
  fprintf(fp, "bias=%lg\n", bias_);
  fprintf(fp, "columns=%d\n", columns_);

  fprintf(fp, "\nweights:\n");
  for (int i = 0; i < columns_; i++) {
    fprintf(fp, "%lg\n", w_[i]);
  }
}
