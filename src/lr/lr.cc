// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//

#include <string>

#include "common/blas-decl.h"
#include "common/hash-entry.h"
#include "common/lbfgs.h"
#include "common/line-reader.h"
#include "lr/lr.h"

LRModel::LRModel() {
  eps_ = 1e-6;
  l1_c_ = 0.0;
  l2_c_ = 0.0;
  positive_weight_ = 1.0;
  ftrl_alpha_ = 0.001;
  ftrl_beta_ = 1.0;
  ftrl_round_ = 1;
  bias_ = 1.0;
  columns_ = 0;
  w_ = NULL;
  ftrl_zn_ = NULL;
}

LRModel::~LRModel() {
  Clear();
}

void LRModel::Clear() {
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

static double LRLoss(
  void* instance,
  const int n,
  const double* w,
  double* g,
  const double step) {
  static const double ZERO = 0.0;

  LossParameter* parameter = (LossParameter*)instance;
  const Problem* problem = parameter->problem;
  LRModel* model = parameter->model;

  int rows = problem->rows();
  int columns = model->columns();
  double positive_weight = model->positive_weight();
  double bias = model->bias();
  double p, gg;
  double fx = 0.0, weighted_rows = 0.0;

  dcopy(columns, &ZERO, 0, g, 1);

  for (int i = 0; i < rows; i++) {
    p = model->Predict(problem->x(i));
    if (problem->y(i) == 1.0) {
      weighted_rows += positive_weight;
      fx -= positive_weight * log(p);
      gg = positive_weight * (p - 1.0);
    } else {
      weighted_rows += 1.0;
      fx -= log(1 - p);
      gg = p;
    }

    // gradient
    for (const FeatureNode* xi = problem->x(i); xi->index != -1; xi++) {
      g[xi->index - 1] += (gg * xi->value);
    }
    if (bias > 0.0) {
      g[columns - 1] += (gg * bias);
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
  Clear();

  columns_ = problem.columns();
  if (bias_ > 0.0) {
    columns_++;
  }

  LossParameter loss_param = {&problem, this};
  lbfgs_parameter_t param;
  lbfgs_default_parameter(&param);
  param.epsilon = eps_;
  if (l1_c_ != 0.0) {
    param.orthantwise_c = l1_c_ / (double)problem.rows();
    param.orthantwise_start = 0;
    param.orthantwise_end = columns_;
    param.linesearch = LBFGS_LINESEARCH_BACKTRACKING_WOLFE;
  }

  w_ = vecalloc(columns_);
  int ret = lbfgs(columns_, w_, 0, &LRLoss, &LRProgress, &loss_param, &param);
  Log("Optimization terminated with status code = %d.\n\n", ret);
}

void LRModel::TrainFTRL(const Problem& problem) {
  // Don't clear previous context
  if (ftrl_zn_ == NULL) {
    Clear();

    columns_ = problem.columns();
    if (bias_ > 0.0) {
      columns_++;
    }

    w_ = vecalloc(columns_);
    ftrl_zn_ = vecalloc(columns_ * 2);
    // ftrl_zn_[2i + 0] is z[i]
    // ftrl_zn_[2i + 1] is n[i]
  }

  int rows = problem.rows();
  for (int i = 0; i < rows; i++) {
    if (i > 0 && i % 10000 == 0) {
      Log("Updated %d samples.\n", i);
    }
    UpdateFTRL(problem.y(i), problem.x(i));
  }
  Log("Updated %d samples.\n", rows);
}

void LRModel::UpdateFTRL(double y, const FeatureNode* x) {
  double* ni, sqrt_ni, new_ni, *zi, sign_zi;
  double p, gg, gi, sigma_i;
  int i;

  for (int times = 0; times < (int)positive_weight_; times++) {
    for (const FeatureNode* xi = x; xi->index != -1; xi++) {
      i = xi->index - 1;
      ni = ftrl_zn_ + 2 * i;
      zi = ni + 1;

      sqrt_ni = sqrt(*ni);
      sign_zi = *zi < 0.0 ? -1.0 : 1.0;
      if (*zi * sign_zi <= l1_c_) {  // fabs(*zi) <= l1_c_
        w_[i] = 0.0;
      } else {
        w_[i] = (sign_zi * l1_c_ - *zi)
                / ((ftrl_beta_ + sqrt_ni) / ftrl_alpha_ + l2_c_);
      }
    }
    if (bias_ > 0.0) {
      i = columns_ - 1;
      ni = ftrl_zn_ + 2 * i;
      zi = ni + 1;

      sqrt_ni = sqrt(*ni);
      sign_zi = *zi < 0.0 ? -1.0 : 1.0;
      if (*zi * sign_zi <= l1_c_) {  // fabs(*zi) <= l1_c_
        w_[i] = 0.0;
      } else {
        w_[i] = (sign_zi * l1_c_ - *zi)
                / ((ftrl_beta_ + sqrt_ni) / ftrl_alpha_ + l2_c_);
      }
    }

    p = Predict(x);
    if (y == 1.0) {
      gg = p - 1.0;
    } else {
      gg = p;
    }

    for (const FeatureNode* xi = x; xi->index != -1; xi++) {
      i = xi->index - 1;
      ni = ftrl_zn_ + 2 * i;
      zi = ni + 1;

      gi = gg * xi->value;
      new_ni = *ni + gi * gi;
      sigma_i = (sqrt(new_ni) - sqrt(*ni)) / ftrl_alpha_;
      *zi += gi - sigma_i * w_[i];
      *ni = new_ni;
    }
    if (bias_ > 0.0) {
      i = columns_ - 1;
      ni = ftrl_zn_ + 2 * i;
      zi = ni + 1;

      gi = gg * bias_;
      new_ni = *ni + gi * gi;
      sigma_i = (sqrt(new_ni) - sqrt(*ni)) / ftrl_alpha_;
      *zi += gi - sigma_i * w_[i];
      *ni = new_ni;
    }
  }
}

void LRModel::Train(const Problem& problem, int mode) {
  if (mode == 0) {
    TrainLBFGS(problem);
  } else {
    for (int r = 0; r < ftrl_round_; r++) {
      TrainFTRL(problem);
    }
  }
}

double LRModel::Predict(const FeatureNode* x) const {
  double wx = 0.0;
  for (; x->index != -1; x++) {
    wx += w_[x->index - 1] * x->value;
  }
  if (bias_ > 0.0) {
    wx += bias_ * w_[columns_ - 1];
  }
  return sigmoid(wx);
}

struct PredictFileProcArg {
  const LRModel* model;
  FILE* fout;
  int dimension;
};

static void PredictFileProc(
  int with_label,
  int sort_x_by_index,
  void* callback_arg,
  double y,
  int sample_max_column,
  FeatureNodeVector* x,
  int error_flag) {
  const LRModel* model = ((PredictFileProcArg*)callback_arg)->model;
  FILE* fout = ((PredictFileProcArg*)callback_arg)->fout;

  if (error_flag == Success) {
    fprintf(fout, "%lg", model->Predict(&(*x)[0]));
  }

  fprintf(fout, "\n");
}

static void PredictFileHashProc(
  void* callback_arg,
  const std::string& name,
  int* hashed_index) {
  int dimension = ((PredictFileProcArg*)callback_arg)->dimension;
  *hashed_index = (unsigned int)HashString(name) % dimension + 1;
}

void LRModel::PredictFile(FILE* fin, FILE* fout, int with_label) const {
  PredictFileProcArg callback_arg = {this, fout, 0};
  ::ForeachFeatureNode(fin, with_label, 0, &callback_arg, &PredictFileProc);
}

void LRModel::PredictHashFile(FILE* fin, FILE* fout,
                              int with_label, int dimension) const {
  PredictFileProcArg callback_arg = {this, fout, dimension};
  ::ForeachFeatureNode_Hash(fin, with_label, 0, &callback_arg,
                            &PredictFileProc, &PredictFileHashProc);
}

void LRModel::Load(FILE* fp) {
  fscanf(fp, "eps=%lg\n", &eps_);
  fscanf(fp, "l1_c=%lg\n", &l1_c_);
  fscanf(fp, "l2_c=%lg\n", &l2_c_);
  fscanf(fp, "positive_weight=%lg\n", &positive_weight_);
  fscanf(fp, "ftrl_alpha=%lg\n", &ftrl_alpha_);
  fscanf(fp, "ftrl_beta=%lg\n", &ftrl_beta_);
  fscanf(fp, "ftrl_round=%d\n", &ftrl_round_);
  fscanf(fp, "bias=%lg\n", &bias_);
  fscanf(fp, "columns=%d\n", &columns_);

  fscanf(fp, "\nweights:\n");
  w_ = vecalloc(columns_);

  char* value;
  LineReader line_reader;
  for (int i = 0; i < columns_; i++) {
    line_reader.ReadLine(fp);
    value = strtok(line_reader.buf, DELIMITER);
    w_[i] = xatod(value);
  }
}

void LRModel::Save(FILE* fp, const FeatureReverseMap* fr_map) const {
  fprintf(fp, "eps=%lg\n", eps_);
  fprintf(fp, "l1_c=%lg\n", l1_c_);
  fprintf(fp, "l2_c=%lg\n", l2_c_);
  fprintf(fp, "positive_weight=%lg\n", positive_weight_);
  fprintf(fp, "ftrl_alpha=%lg\n", ftrl_alpha_);
  fprintf(fp, "ftrl_beta=%lg\n", ftrl_beta_);
  fprintf(fp, "ftrl_round=%d\n", ftrl_round_);
  fprintf(fp, "bias=%lg\n", bias_);
  fprintf(fp, "columns=%d\n", columns_);

  fprintf(fp, "\nweights:\n");
  for (int i = 0; i < columns_; i++) {
    fprintf(fp, "%lg", w_[i]);
    if (fr_map) {
      if (i == columns_ - 1) {
        fprintf(fp, "\tBIAS");
      } else {
        FeatureReverseMapCII ii = fr_map->equal_range(i + 1);
        for (FeatureReverseMapCI it = ii.first; it != ii.second; ++it) {
          fprintf(fp, "\t%s", it->second.c_str());
        }
      }
    }
    fprintf(fp, "\n");
  }
}
