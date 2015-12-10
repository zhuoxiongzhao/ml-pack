// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// map non-LIBSVM sample files to LIBSVM format with a feature map
//

#include <algorithm>
#include <map>
#include <string>

#include "core/line-reader.h"
#include "core/problem.h"

typedef std::map<std::string, int> FeatureMap;

std::string feature_map_filename = "feature-map";
int with_label = 1;
int sort_feature = 1;
std::string mapped_sample_filename = "-";

void Process(FILE* fin, FILE* fout, const FeatureMap& feature_index_map) {
  LineReader line_reader;
  int i = 0;
  char* endptr;
  char* label;
  char* index;
  char* value;
  char* feature_begin;
  int error_flag;
  FeatureNodeVector x;
  FeatureNode node;

  while (line_reader.ReadLine(fin) != NULL) {
    if (with_label) {
      // label
      label = strtok(line_reader.buf, DELIMITER);
      if (label == NULL) {
        // empty line
        goto next_line;
      }
      fprintf(fout, "%s", label);
      feature_begin = NULL;
    } else {
      feature_begin = line_reader.buf;
    }

    // features
    error_flag = 0;
    x.clear();
    for (;;) {
      index = strtok(feature_begin, DELIMITER);
      feature_begin = NULL;
      if (index == NULL) {
        break;
      }

      value = strrchr(index, ':');
      if (value) {
        if (value == index) {
          Error("line %d, feature name is empty.\n", i + 1);
          error_flag = 1;
        }
        *value = '\0';
        value++;
        node.value = (float)strtod(value, &endptr);
        if (*endptr != '\0') {
          Error("line %d, feature value error \"%s\".\n", i + 1, value);
          error_flag = 1;
        }
      } else {
        node.value = 1.0;
      }

      if (error_flag == 0) {
        FeatureMap::const_iterator it =
          feature_index_map.find(std::string(index));
        if (it != feature_index_map.end()) {
          node.index = it->second;
          x.push_back(node);
        }
      }
    }

    if (!x.empty()) {
      if (sort_feature) {
        std::sort(x.begin(), x.end(), FeatureNodeLess());
      }
      if (with_label) {
        fprintf(fout, " %d:%g", x[0].index, x[0].value);
      } else {
        fprintf(fout, "%d:%g", x[0].index, x[0].value);
      }
      for (size_t j = 1; j < x.size(); j++) {
        fprintf(fout, " %d:%g", x[j].index, x[j].value);
      }
    }

next_line:
    fprintf(fout, "\n");
    i++;
  }
}

void LoadFeatureMap(FILE* fp, FeatureMap* feature_index_map) {
  LineReader line_reader;
  std::string name;
  int index;
  char* line;

  while (line_reader.ReadLine(fp) != NULL) {
    line = strtok(line_reader.buf, DELIMITER);
    if (line == NULL) {
      // empty line
      continue;
    }

    name = line;
    // index starts from 1
    index = (int)feature_index_map->size() + 1;
    (*feature_index_map)[name] = index;
  }
}

void Usage() {
  fprintf(stderr,
          "Usage: map-sample [options] SAMPLE_FILE\n"
          "  SAMPLE_FILE: input sample filename, \"-\" denotes stdin.\n"
          "\n"
          "  Options:\n"
          "    -f FEATURE_MAP_FILE\n"
          "      The input feature map filename.\n"
          "      Default is \"%s\".\n"
          "    -l WITH_LABEL(0 or 1)\n"
          "      Whether SAMPLE_FILE contains labels.\n"
          "      Default is \"%d\".\n"
          "    -s SORT_FEATURE_BY_INDEX(0 or 1)\n"
          "      Whether sort features by index.\n"
          "      Default is \"%d\".\n"
          "    -o MAPPED_SAMPLE_FILE\n"
          "      Default is \"%s\".\n",
          feature_map_filename.c_str(),
          with_label,
          sort_feature,
          mapped_sample_filename.c_str());
  exit(1);
}

int main(int argc, char** argv) {
  if (argc == 1) {
    Usage();
  }

  int i = 1;
  for (;;) {
    std::string s = argv[i];
    if (s == "-h" || s == "-help" || s == "--help") {
      Usage();
    }

    if (s == "-f") {
      CHECK_MISSING_ARG(argc, argv, i, Usage());
      feature_map_filename = argv[i + 1];
      COMSUME_2_ARG(argc, argv, i);
    } else if (s == "-l") {
      CHECK_MISSING_ARG(argc, argv, i, Usage());
      with_label = xatoi(argv[i + 1]);
      COMSUME_2_ARG(argc, argv, i);
    } else if (s == "-s") {
      CHECK_MISSING_ARG(argc, argv, i, Usage());
      sort_feature = xatoi(argv[i + 1]);
      COMSUME_2_ARG(argc, argv, i);
    } else if (s == "-o") {
      CHECK_MISSING_ARG(argc, argv, i, Usage());
      mapped_sample_filename = argv[i + 1];
      COMSUME_2_ARG(argc, argv, i);
    } else {
      i++;
    }
    if (i == argc) {
      break;
    }
  }

  if (argc == 1) {
    Usage();
  }

  FeatureMap feature_index_map;
  {
    ScopedFile fp(feature_map_filename.c_str(), ScopedFile::Read);
    LoadFeatureMap(fp, &feature_index_map);
  }
  {
    ScopedFile fin(argv[1], ScopedFile::Read);
    ScopedFile fout(mapped_sample_filename.c_str(), ScopedFile::Write);
    Log("Mapping \"%s\" to \"%s\"...\n",
        argv[1], mapped_sample_filename.c_str());
    Process(fin, fout, feature_index_map);
    Log("Done.\n\n");
  }
  return 0;
}
