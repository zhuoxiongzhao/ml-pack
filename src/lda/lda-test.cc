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
  prob.push_back(1.0);
  prob.push_back(2.0);
  prob.push_back(3.0);
  prob.push_back(4.0);
  prob.push_back(5.0);

  Alias alias;
  alias.Construct(prob);

  std::vector<int> count(alias.n());
  for (int i = 0; i < 10000; i++) {
    count[alias.Sample()]++;
  }
  for (int i = 0; i < alias.n(); i++) {
    printf("%d\n", count[i]);
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
  SparseLDASampler model;
  model.LoadCorpus(fp, 1);
  model.K() = 3;
  model.alpha() = 0.1;
  model.beta() = 0.1;
  model.total_iteration() = 100;
  model.hp_opt() = 0;
  model.storage_type() = kSparseHist;
  model.Train();
  model.SaveModel(TEST_DATA_DIR"/yahoo");
}

int main() {
  // TestAlias();
  TestSimple();
  TestYahoo();
  return 0;
}
