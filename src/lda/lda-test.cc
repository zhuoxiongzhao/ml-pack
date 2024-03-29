// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// lda tests
//

#include "common/x.h"
#include "lda/alias.h"
#include "lda/sampler.h"

#if defined _WIN32
#define TEST_DATA_DIR "../src/lda-test-data"
#else
#define TEST_DATA_DIR "lda-test-data"
#endif

void TestAlias() {
  std::vector<double> prob;
  prob.push_back(0.01);
  prob.push_back(0.07);
  prob.push_back(0.13);
  prob.push_back(0.29);
  prob.push_back(0.45);
  prob.push_back(0.05);

  Alias alias;
  alias.Build(prob, 1.0);

  std::vector<int> count(alias.n());
  const int N = 10000;
  for (int i = 0; i < N; i++) {
    count[alias.Sample()]++;
  }
  for (int i = 0; i < alias.n(); i++) {
    printf("%lf\n", count[i] / (double)N);
  }
}

void TestSimple() {
  ScopedFile fp(TEST_DATA_DIR"/simple-train", ScopedFile::Read);
  LightLDASampler model;
  model.LoadCorpus(fp, 0);
  model.K() = 2;
  model.alpha() = 0.1;
  model.beta() = 0.1;
  model.total_iteration() = 100;
  model.hp_opt() = 0;
  model.storage_type() = kSparseHist;
  model.Train();
  model.SaveModel(TEST_DATA_DIR"/simple");
}

void TestYahoo() {
  ScopedFile fp(TEST_DATA_DIR"/yahoo-train", ScopedFile::Read);
  // GibbsSampler model;  //-83246.6/-6.99258
  // SparseLDASampler model;  //-83226.5/-6.99089
  // AliasLDASampler model;  // -83187.8/-6.98764
  LightLDASampler model;  // -83292.9/-6.99647
  model.mh_step() = 16;
  model.LoadCorpus(fp, 0);
  model.K() = 3;
  model.alpha() = 0.1;
  model.beta() = 0.1;
  model.burnin_iteration() = 0;
  model.log_likelihood_interval() = 1;
  model.total_iteration() = 100;
  model.hp_opt() = 0;
  model.storage_type() = kSparseHist;
  model.Train();
  model.SaveModel(TEST_DATA_DIR"/yahoo");
}

void TestNIPS() {
  ScopedFile fp(TEST_DATA_DIR"/nips-train", ScopedFile::Read);
  LightLDASampler model;
  model.mh_step() = 8;
  model.LoadCorpus(fp, 0);
  model.K() = 50;
  model.alpha() = 0.1;
  model.beta() = 0.1;
  model.burnin_iteration() = 0;
  model.log_likelihood_interval() = 666;
  model.total_iteration() = 50;
  model.hp_opt() = 0;
  model.storage_type() = kSparseHist;
  model.Train();
  model.SaveModel(TEST_DATA_DIR"/nips");
}

int main() {
  // TestAlias();
  // TestSimple();
  TestYahoo();
  // TestNIPS();
  return 0;
}
