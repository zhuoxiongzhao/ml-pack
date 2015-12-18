// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//

#include "common/line-reader.h"
#include "common/mt19937ar.h"
#include "common/mt64.h"
#include "lda/model.h"

void LDAModel::LoadCorpus(FILE* fp, int with_id) {
  LineReader line_reader;
  int line_no = 1;
  char* endptr;
  char* doc_id;
  char* word_id;
  char* word_begin;
  Doc doc;
  int id;
  Word word;

  Log("Loading corpus.\n");
  V_ = 0;
  while (line_reader.ReadLine(fp) != NULL) {
    doc.index = (int)words_.size();
    doc.N = 0;

    if (with_id) {
      doc_id = strtok(line_reader.buf, DELIMITER);
      if (doc_id == NULL) {
        Error("line %d, empty line.\n", line_no);
        continue;
      }
      word_begin = NULL;
    } else {
      word_begin = line_reader.buf;
    }

    for (;;) {
      word_id = strtok(word_begin, DELIMITER);
      word_begin = NULL;
      if (word_id == NULL) {
        break;
      }

      id = (int)strtoll(word_id, &endptr, 10);
      if (*endptr != '\0') {
        Error("line %d, word id error \"%s\".\n", line_no, word_id);
        continue;
      }
      if (id == 0) {
        Error("line %d, word id must start from 1.\n", line_no);
        continue;
      }

      if (id > V_) {
        V_ = id;
      }
      word.v = id - 1;
      words_.push_back(word);
      doc.N++;
    }

    if (doc.N) {
      if (with_id) {
        doc_ids_.push_back(doc_id);
      }
      docs_.push_back(doc);
    }
  }

  M_ = (int)docs_.size();
  Log("Loaded %d documents with a %d-size vocabulary.\n", M_, V_);
}

void LDAModel::LoadModel(const std::string& prefix) {
  std::string filename;
  Log("Loading model.\n");
  {
    filename = prefix + "-stat";
    ScopedFile fp(filename.c_str(), ScopedFile::Read);
    fscanf(fp, "M=%d\n", &M_);
    fscanf(fp, "V=%d\n", &V_);
    fscanf(fp, "K=%d\n", &K_);
  }
  {
    filename = prefix + "-doc-topic";
    ScopedFile fp(filename.c_str(), ScopedFile::Read);
    for (int m = 0; m < M_; m++) {
      for (int k = 0; k < K_ - 1; k++) {
        fscanf(fp, "%lg ", &theta_mk_[m][k]);
      }
      fscanf(fp, "%lg\n", &theta_mk_[m][K_ - 1]);
    }
  }
  {
    filename = prefix + "-topic-word";
    ScopedFile fp(filename.c_str(), ScopedFile::Read);
    for (int k = 0; k < K_; k++) {
      for (int v = 0; v < V_ - 1; v++) {
        fscanf(fp, "%lg ", &phi_kv_[k][v]);
      }
      fscanf(fp, "%lg\n", &phi_kv_[k][V_ - 1]);
    }
  }
  Log("Done.\n");
}

void LDAModel::SaveModel(const std::string& prefix) const {
  int m, v, k;
  std::string filename;
  Log("Saving model.\n");
  {
    filename = prefix + "-stat";
    ScopedFile fp(filename.c_str(), ScopedFile::Write);
    fprintf(fp, "M=%d\n", M_);
    fprintf(fp, "V=%d\n", V_);
    fprintf(fp, "K=%d\n", K_);
  }
  {
    filename = prefix + "-doc-topic";
    ScopedFile fp(filename.c_str(), ScopedFile::Write);
    for (m = 0; m < M_; m++) {
      for (k = 0; k < K_ - 1; k++) {
        fprintf(fp, "%lg ", theta_mk_[m][k]);
      }
      fprintf(fp, "%lg\n", theta_mk_[m][K_ - 1]);
    }
  }
  {
    filename = prefix + "-topic-word";
    ScopedFile fp(filename.c_str(), ScopedFile::Write);
    for (k = 0; k < K_; k++) {
      for (v = 0; v < V_ - 1; v++) {
        fprintf(fp, "%lg ", phi_kv_[k][v]);
      }
      fprintf(fp, "%lg\n", theta_mk_[k][V_ - 1]);
    }
  }
  Log("Done.\n");
}

void LDAModel::InitializeModel() {
  int m, n, v, k;
  const Doc* doc;
  Word* word;
  int* N_m;

  if (K_ == 0) {
    Error("Please set number topics.\n");
    return;
  }

  Log("Initializing model with %d topics.\n", K_);
  N_mk_.Init(M_, K_);
  N_k_.resize(K_);
  N_kv_.Init(K_, V_);

  theta_mk_.Init(M_, K_);
  phi_kv_.Init(K_, V_);

  if (total_alpha_ <= 0.0) {
    double avg_doc_len = (double)words_.size() / docs_.size();
    alpha_k_.resize(K_, avg_doc_len / K_);
    total_alpha_ = avg_doc_len;
  } else {
    alpha_k_.resize(K_, total_alpha_);
    total_alpha_ = total_alpha_ * K_;
  }

  if (beta_ <= 0.0) {
    beta_ = 0.01;
  }

  for (m = 0; m < M_; m++) {
    doc = &docs_[m];
    word = &words_[doc->index];
    N_m = N_mk_[m];
    for (n = 0; n < doc->N; n++, word++) {
      v = word->v;
      k = (int)(genrand_int32() % K_);
      word->k = k;
      N_m[k]++;
      N_kv_[k][v]++;
      N_k_[k]++;
    }
  }

  if (total_iteration_ == 0) {
    total_iteration_ = 50;
  }
  iteration_ = 1;
  Log("Done.\n");
}

void LDAModel::OptimizeHyper() {
  // TODO(yafei)
}

void LDAModel::CollectThetaPhi() {
  int m, v, k;
  const Doc* doc;
  double* theta_m, * phi_k;
  int* N_m, * N_k;
  const double vbeta = V_ * beta_;

  for (m = 0; m < M_; m++) {
    doc = &docs_[m];
    theta_m = theta_mk_[m];
    N_m = N_mk_[m];
    for (k = 0; k < K_; k++) {
      theta_m[k] = 0.0;
    }
    for (k = 0; k < K_; k++) {
      theta_m[k] += (N_m[k] + alpha_k_[k]) / (doc->N + total_alpha_);
    }
  }

  for (k = 0; k < K_; k++) {
    phi_k = phi_kv_[k];
    N_k = N_kv_[k];
    for (v = 0; v < V_; v++) {
      phi_k[v] = 0.0;
    }
    for (v = 0; v < V_; v++) {
      phi_k[v] += (N_k[v] + beta_) / (N_k_[k] + vbeta);
    }
  }
}

double LDAModel::LogLikelyhood() const {
  int m, n, v, k;
  const Doc* doc;
  const Word* word;
  const int* N_m;
  double sum = 0.0, word_sum;
  const double vbeta = V_ * beta_;

  for (m = 0; m < M_; m++) {
    doc = &docs_[m];
    word = &words_[doc->index];
    N_m = N_mk_[m];
    for (n = 0; n < doc->N; n++, word++) {
      v = word->v;
      k = word->k;
      word_sum = 0.0;
      for (k = 0; k < K_; k++) {
        word_sum += (N_m[k] + alpha_k_[k])
                    * (N_kv_[k][v] + beta_)
                    / (N_k_[k] + vbeta);
      }
      word_sum /= (doc->N + total_alpha_);
      sum += log(word_sum);
    }
  }

  return sum;
}

void PlainGibbsSampler::InitializeModel() {
  LDAModel::InitializeModel();
  topic_cdf_.resize(K_);
}

void PlainGibbsSampler::SampleCorpus() {
  int m, n, v, k;
  const Doc* doc;
  Word* word;
  int* N_m;
  double talpha, r;
  const double vbeta = V_ * beta_;

  for (m = 0; m < M_; m++) {
    doc = &docs_[m];
    word = &words_[doc->index];
    N_m = N_mk_[m];
    for (n = 0; n < doc->N; n++, word++) {
      v = word->v;
      k = word->k;
      talpha = doc->N - 1 + total_alpha_;

      N_m[k]--;
      N_kv_[k][v]--;
      N_k_[k]--;

      topic_cdf_[0] = 0.0;
      for (k = 0; k < K_ - 1; k++) {
        topic_cdf_[k] += (N_kv_[k][v] + beta_)
                         / (N_k_[k] + vbeta)
                         * (N_m[k] + alpha_k_[k])
                         / talpha;
        topic_cdf_[k + 1] = topic_cdf_[k];
      }
      topic_cdf_[K_ - 1] += (N_kv_[K_ - 1][v] + beta_)
                            / (N_k_[K_ - 1] + vbeta)
                            * (N_m[K_ - 1] + alpha_k_[K_ - 1])
                            / talpha;

      r = genrand64_real3() * topic_cdf_[K_ - 1];
      for (k = 0; k < K_; k++) {
        if (topic_cdf_[k] >= r) {
          break;
        }
      }

      N_m[k]++;
      N_kv_[k][v]++;
      N_k_[k]++;
      word->k = k;
    }
  }
}

void PlainGibbsSampler::Train() {
  InitializeModel();
  for (iteration_ = 1; iteration_ <= total_iteration_; iteration_++) {
    Log("Iteration %d started.\n", iteration_);
    SampleCorpus();
    CollectThetaPhi();
    Log("LogLikelyhood=%lg\n", LogLikelyhood());
    Log("Iteration %d finished.\n", iteration_);
  }
}
