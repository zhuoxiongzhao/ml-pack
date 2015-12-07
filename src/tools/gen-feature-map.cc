// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// map non-LIBSVM sample files to LIBSVM format and generate a feature map
//

#include <map>
#include <string>

#include "common/line-reader.h"
#include "common/problem.h"
#include "common/x.h"

typedef std::map<std::string, int> FeatureMap;

std::string feature_map_filename = "feature-map";
int with_label = 1;
int threshold = 0;

void Process(FILE* fp, FeatureMap* feature_count_map) {
  LineReader line_reader;
  int i = 0;
  char* endptr;
  char* label;
  char* index;
  char* value;
  char* feature_begin;
  std::string name;

  while (line_reader.ReadLine(fp) != NULL) {
    if (with_label) {
      // label
      label = strtok(line_reader.buf, DELIMITER);
      if (label == NULL) {
        // empty line
        goto next_line;
      }
      feature_begin = NULL;
    } else {
      feature_begin = line_reader.buf;
    }

    // features
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
          exit(3);
        }
        *value = '\0';
        value++;
        strtod(value, &endptr);
        if (*endptr != '\0') {
          Error("line %d, feature value error \"%s\".\n", i + 1, value);
          exit(4);
        }
      }

      name = index;
      (*feature_count_map)[name]++;
    }

next_line:
    i++;
  }
}

void SaveFeatureMap(FILE* fp, const FeatureMap& feature_count_map) {
  Log("Writing to \"%s\"...\n", feature_map_filename.c_str());
  FeatureMap::const_iterator first = feature_count_map.begin();
  FeatureMap::const_iterator last = feature_count_map.end();
  for (; first != last; ++first) {
    if (first->second > threshold) {
      fprintf(fp, "%s\t%d\n", first->first.c_str(), first->second);
    }
  }
  Log("Done.\n\n");
}

void Usage() {
  fprintf(stderr,
          "Usage: gen-feature-map [options] SAMPLE_FILE\n"
          "  SAMPLE_FILE: input sample filename, \"-\" denotes stdin.\n"
          "\n"
          "  Options:\n"
          "    -f FEATURE_MAP_FILE\n"
          "      The output feature map filename.\n"
          "      Default is \"%s\".\n"
          "    -l WITH_LABEL(0 or 1)\n"
          "      Whether SAMPLE_FILE contains labels.\n"
          "      Default is \"%d\".\n"
          "    -t THRESHOLD\n"
          "      Keep features whose frequency "
          "are larger than this threshold.\n"
          "      Default is \"%d\".\n",
          feature_map_filename.c_str(),
          with_label,
          threshold);
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
    } else if (s == "-t") {
      if (i + 1 == argc) {
        MISSING_ARG(argc, argv, i);
        Usage();
      }
      threshold = xatoi(argv[i + 1]);
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

  FeatureMap feature_count_map;
  {
    ScopedFile fp(argv[1], ScopedFile::Read);
    Log("Processing \"%s\"...\n", argv[i]);
    Process(fp, &feature_count_map);
    Log("Done.\n\n");
  }
  {
    ScopedFile fp(feature_map_filename.c_str(), ScopedFile::Write);
    SaveFeatureMap(fp, feature_count_map);
  }
  return 0;
}
