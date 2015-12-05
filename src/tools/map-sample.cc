// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// map non-LIBSVM sample files to LIBSVM format with a feature map
//

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "common/line-reader.h"
#include "common/problem.h"

typedef std::map<std::string, int> FeatureMap;

std::string feature_map_filename = "feature-map";
int with_label = 1;

void Process(FILE* fin, FILE* fout, const FeatureMap& feature_index_map) {
  LineReader line_reader;
  int i = 0;
  char* endptr;
  char* label;
  char* index;
  char* value;
  std::vector<FeatureNode> x;
  FeatureNode feature;

  while (line_reader.ReadLine(fin) != NULL) {
    // label
    label = strtok(line_reader.buf, DELIMITER);
    if (label == NULL) {
      // empty line
      continue;
    }

    fprintf(fout, "%s", label);

    // features
    x.clear();
    for (;;) {
      index = strtok(NULL, DELIMITER);
      if (index == NULL) {
        break;
      }

      value = strrchr(index, ':');
      if (value) {
        if (value == index) {
          Error("line %d, feature name is empty.\n", i + 1);
          exit(2);
        }
        *value = '\0';
        value++;
        feature.value = strtod(value, &endptr);
        if (*endptr != '\0') {
          Error("line %d, feature value error \"%s\".\n", i + 1, value);
          exit(3);
        }
      } else {
        feature.value = 1.0;
      }

      if (feature.value > -EPSILON && feature.value < EPSILON) {
        continue;
      }

      FeatureMap::const_iterator it =
        feature_index_map.find(std::string(index));
      if (it != feature_index_map.end()) {
        feature.index = it->second;
        x.push_back(feature);
      }
    }

    std::sort(x.begin(), x.end(), FeatureNodeLess());
    for (size_t j = 0; j < x.size(); j++) {
      fprintf(fout, " %d:%g", x[j].index, x[j].value);
    }
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
    index = (int)feature_index_map->size() + 1;
    (*feature_index_map)[name] = index;
  }
}

void Usage() {
  fprintf(stderr,
          "Usage: map-sample [options] SAMPLE_FILE1 [SAMPLE_FILE2] ...\n"
          "  SAMPLE_FILE: input sample filename.\n"
          "    A postfix \".libsvm\" will be added to SAMPLE_FILE.\n"
          "\n"
          "  Options:\n"
          "    -f FEATURE_MAP_FILENAME\n"
          "      The input feature map filename.\n"
          "      Default is \"%s\".\n"
          "    -l WITH_LABEL(0 or 1)\n"
          "      Whether SAMPLE_FILE contains labels.\n"
          "      Default is \"%d\".\n",
          feature_map_filename.c_str(),
          with_label);
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
      if (i + 1 == argc) {
        MISSING_ARG(argc, argv, i);
        Usage();
      }
      feature_map_filename = argv[i + 1];
      COMSUME_2_ARG(argc, argv, i);
    } else if (s == "-l") {
      if (i + 1 == argc) {
        MISSING_ARG(argc, argv, i);
        Usage();
      }
      with_label = xatoi(argv[i + 1]);
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

  for (i = 1; i < argc; i++) {
    std::string filename = argv[i];
    filename += ".libsvm";
    ScopedFile fin(argv[i], ScopedFile::Read);
    ScopedFile fout(filename.c_str(), ScopedFile::Write);
    Log("Mapping \"%s\" to \"%s.libsvm\"...\n", argv[i], argv[i]);
    Process(fin, fout, feature_index_map);
  }

  return 0;
}
