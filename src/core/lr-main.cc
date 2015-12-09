// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// lr train and predict
//

#include <string>

#include "common/metric.h"
#include "core/lr.h"

// 0, train
// 1, predict
int action = 0;
// train
std::string model_filename = "model";
double eps = 1e-6;
double l1_c = 1.0;
double l2_c = 0.0;
double positive_weight = 1.0;
double bias = 1.0;
double testing_portion = 0.0;
int nfold = 0;
// predict
int with_label = 1;
std::string predict_filename = "-";

void Usage() {
  fprintf(stderr,
          "Usage: lr-main action [options] SAMPLE_FILE\n"
          "  SAMPLE_FILE: input sample filename.\n"
          "\n"
          "  Action:\n"
          "    train\n"
          "    predict\n");
  exit(1);
}

void SubUsage() {
  if (action == 0) {
    fprintf(stderr,
            "Usage: lr-main train [options] SAMPLE_FILE\n"
            "  SAMPLE_FILE: input sample filename, \"-\" denotes stdin.\n"
            "\n"
            "  Options:\n"
            "    -o MODEL_FILE\n"
            "      Output model filename.\n"
            "      Default is \"%s\".\n"
            "    -e EPSILON\n"
            "      Termination criteria.\n"
            "      Default is \"%lg\".\n"
            "    -l1 C\n"
            "      L1 regularization coefficient.\n"
            "      Default is \"%lg\".\n"
            "    -l2 C\n"
            "      L2 regularization coefficient.\n"
            "      Default is \"%lg\".\n"
            "    -pw POSITIVE_WEIGHT\n"
            "      Weights of all positive samples.\n"
            "      Default is \"%lg\".\n"
            "    -b BIAS\n"
            "      Value of the bias term(no bias term if BIAS <= 0).\n"
            "      Default is \"%lg\".\n"
            "    -t TESTING_PORTION\n"
            "      Portion of testing set, "
            "enabled if 0 < TESTING_PORTION < 1.\n"
            "      Default is \"%lg\".\n"
            "    -cv N\n"
            "      N-fold cross validation, "
            "enabled if N > 0 and testing set is disabled\n"
            "      Default is \"%d\".\n",
            model_filename.c_str(),
            eps, l1_c, l2_c, positive_weight,
            bias, testing_portion, nfold);
  } else {
    fprintf(stderr,
            "Usage: lr-main predict [options] SAMPLE_FILE\n"
            "  SAMPLE_FILE: input sample filename, \"-\" denotes stdin.\n"
            "\n"
            "  Options:\n"
            "    -m MODEL_FILE\n"
            "      Input model filename.\n"
            "      Default is \"%s\".\n"
            "    -l WITH_LABEL(0 or 1)\n"
            "      Whether SAMPLE_FILE contains labels.\n"
            "      Default is \"%d\".\n"
            "    -o PREDICT_FILE\n"
            "      Output predict filename, \"-\" denotes stdout.\n"
            "      Default is \"%s\".\n",
            model_filename.c_str(),
            with_label,
            predict_filename.c_str());
  }
  exit(1);
}

void Train(int argc, char** argv) {
  if (argc == 1) {
    SubUsage();
  }

  int i = 1;
  for (;;) {
    std::string s = argv[i];
    if (s == "-h" || s == "-help" || s == "--help") {
      SubUsage();
    }

    if (s == "-o") {
      if (i + 1 == argc) {
        MISSING_ARG(argc, argv, i);
        SubUsage();
      }
      model_filename = argv[i + 1];
      COMSUME_2_ARG(argc, argv, i);
    } else if (s == "-e") {
      if (i + 1 == argc) {
        MISSING_ARG(argc, argv, i);
        SubUsage();
      }
      eps = xatod(argv[i + 1]);
      COMSUME_2_ARG(argc, argv, i);
    } else if (s == "-l1") {
      if (i + 1 == argc) {
        MISSING_ARG(argc, argv, i);
        SubUsage();
      }
      l1_c = xatod(argv[i + 1]);
      COMSUME_2_ARG(argc, argv, i);
    } else if (s == "-l2") {
      if (i + 1 == argc) {
        MISSING_ARG(argc, argv, i);
        SubUsage();
      }
      l2_c = xatod(argv[i + 1]);
      COMSUME_2_ARG(argc, argv, i);
    } else if (s == "-pw") {
      if (i + 1 == argc) {
        MISSING_ARG(argc, argv, i);
        SubUsage();
      }
      positive_weight = xatod(argv[i + 1]);
      COMSUME_2_ARG(argc, argv, i);
    } else if (s == "-b") {
      if (i + 1 == argc) {
        MISSING_ARG(argc, argv, i);
        SubUsage();
      }
      bias = xatod(argv[i + 1]);
      COMSUME_2_ARG(argc, argv, i);
    } else if (s == "-t") {
      if (i + 1 == argc) {
        MISSING_ARG(argc, argv, i);
        SubUsage();
      }
      testing_portion = xatod(argv[i + 1]);
      COMSUME_2_ARG(argc, argv, i);
    } else if (s == "-cv") {
      if (i + 1 == argc) {
        MISSING_ARG(argc, argv, i);
        SubUsage();
      }
      nfold = xatoi(argv[i + 1]);
      COMSUME_2_ARG(argc, argv, i);
    } else {
      i++;
    }
    if (i == argc) {
      break;
    }
  }

  if (argc == 1) {
    SubUsage();
  }

  Problem problem;
  {
    ScopedFile fp(argv[1], ScopedFile::Read);
    problem.LoadText(fp, bias);
    //problem.LoadHashText(fp, bias, 1024);
  }

  LRModel model;
  model.eps() = eps;
  model.l1_c() = l1_c;
  model.l2_c() = l2_c;
  model.positive_weight() = positive_weight;

  if (testing_portion > 0.0 && testing_portion < 1.0) {
    Problem training, testing;

    Log("Splitting training and testing samples.\n");
    problem.GenerateTrainingTesting(&training, &testing, testing_portion);
    Log("%d samples are used to train.\n", training.rows());
    Log("%d samples are used to test.\n", testing.rows());
    Log("Done.\n\n");

    model.TrainLBFGS(training);
    {
      Log("Writing to \"%s\"...\n", model_filename.c_str());
      ScopedFile fp(model_filename.c_str(), ScopedFile::Write);
      model.Save(fp);
      Log("Done.\n\n");
    }

    std::vector<double> pred;
    pred.reserve(testing.rows());
    for (int j = 0; j < testing.rows(); j++) {
      pred.push_back(model.Predict(testing.x(j)));
    }

    BinaryClassificationMetric metric;
    Evaluate(pred, testing.y(), &metric);
  } else if (nfold > 0) {
    Problem* nfold_training = new Problem[nfold];
    Problem* nfold_testing = new Problem[nfold];

    Log("Generating %d fold samples.\n", nfold);
    problem.GenerateNFold(nfold_training, nfold_testing, nfold);
    Log("Done.\n\n");

    for (int i = 0; i < nfold; i++) {
      Log("---------------Fold %d---------------\n", i);

      char buf[16];
      snprintf(buf, sizeof(buf), "%d", i);
      std::string model_filename_i = model_filename + buf;

      model.TrainLBFGS(nfold_training[i]);
      {
        Log("Writing to \"%s\"...\n", model_filename_i.c_str());
        ScopedFile fp(model_filename_i.c_str(), ScopedFile::Write);
        model.Save(fp);
        Log("Done.\n\n");
      }

      std::vector<double> pred;
      pred.reserve(nfold_testing[i].rows());
      for (int j = 0; j < nfold_testing[i].rows(); j++) {
        pred.push_back(model.Predict(nfold_testing[i].x(j)));
      }

      BinaryClassificationMetric metric;
      Evaluate(pred, nfold_testing[i].y(), &metric);
    }

    delete [] nfold_training;
    delete [] nfold_testing;
  } else {
    model.TrainLBFGS(problem);
    {
      Log("Writing to \"%s\"...\n", model_filename.c_str());
      ScopedFile fp(model_filename.c_str(), ScopedFile::Write);
      model.Save(fp);
      Log("Done.\n\n");
    }
  }
}

void Predict(int argc, char** argv) {
  if (argc == 1) {
    SubUsage();
  }

  int i = 1;
  for (;;) {
    std::string s = argv[i];
    if (s == "-h" || s == "-help" || s == "--help") {
      SubUsage();
    }

    if (s == "-m") {
      if (i + 1 == argc) {
        MISSING_ARG(argc, argv, i);
        SubUsage();
      }
      model_filename = argv[i + 1];
      COMSUME_2_ARG(argc, argv, i);
    } else if (s == "-l") {
      if (i + 1 == argc) {
        MISSING_ARG(argc, argv, i);
        SubUsage();
      }
      with_label = xatoi(argv[i + 1]);
      COMSUME_2_ARG(argc, argv, i);
    } else if (s == "-o") {
      if (i + 1 == argc) {
        MISSING_ARG(argc, argv, i);
        SubUsage();
      }
      predict_filename = argv[i + 1];
      COMSUME_2_ARG(argc, argv, i);
    } else {
      i++;
    }
    if (i == argc) {
      break;
    }
  }

  if (argc == 1) {
    SubUsage();
  }

  LRModel model;
  {
    ScopedFile fp(model_filename.c_str(), ScopedFile::Read);
    model.Load(fp);
  }
  {
    ScopedFile fin(argv[1], ScopedFile::Read);
    ScopedFile fout(predict_filename.c_str(), ScopedFile::Write);
    model.Predict(fin, fout, with_label);
  }
}

int main(int argc, char** argv) {
  if (argc == 1) {
    Usage();
  }

  if (strcmp("train", argv[1]) == 0) {
    action = 0;
    COMSUME_1_ARG(argc, argv, 1);
    Train(argc, argv);
  } else if (strcmp("predict", argv[1]) == 0) {
    action = 1;
    COMSUME_1_ARG(argc, argv, 1);
    Predict(argc, argv);
  } else {
    Usage();
  }
  return 0;
}
