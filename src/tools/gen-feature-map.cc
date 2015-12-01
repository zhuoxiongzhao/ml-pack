// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (kimmyzhang@tencent.com)
//
// Convert x-format data to LIBSVM/LIBLINEAR format.
//

#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include "x.h"

typedef std::map<std::string, int> FeatureMap;

bool Process(FILE* fin, FILE* fout,
             FeatureMap* feature_id_map,
             FeatureMap* feature_count_map) {
  static const char* kDelimiter = " \t|\n";
  static const char kDelimiterChar = ' ';

  LineReader line_reader;
  std::vector<FeatureNode> x;
  FeatureNode feature;
  std::string feature_name;
  int i = 0;
  char* endptr;
  char* label;
  char* index;
  char* value;

  while (line_reader.ReadLine(fin) != NULL) {
    label = strtok(line_reader.buf, kDelimiter);
    if (label == NULL) {
      // empty line
      continue;
    }

    fprintf(fout, "%s", label);
    x.clear();

    for (;;) {
      index = strtok(NULL, kDelimiter);
      if (index == NULL)
        break;

      value = strrchr(index, ':');
      if (value) {
        *value = '\0';
        value++;
        feature.value = strtod(value, &endptr);
        if (endptr == value || *endptr != '\0') {
          Debug("line %d, feature value error \"%s\".\n", i + 1, value);
          return false;
        }
      } else {
        feature.value = 1.0;
      }

      feature_name = index;
      feature.index = (*feature_id_map)[feature_name];
      if (feature.index == 0) {
        (*feature_id_map)[feature_name] = (int)feature_id_map->size();
        feature.index = (int)feature_id_map->size();
      }
      (*feature_count_map)[feature_name]++;
      if (endptr == index || *endptr != '\0') {
        Debug("line %d, feature index error \"%s\".\n", i + 1, index);
        return false;
      }

      x.push_back(feature);
    }

    std::sort(x.begin(), x.end(), FeatureNodeLess());
    for (size_t j = 0; j < x.size(); j++) {
      fprintf(fout, "%c%d:%g", kDelimiterChar, x[j].index, x[j].value);
    }
    fprintf(fout, "\n");
    i++;
  }

  return true;
}

int main(int argc, char** argv) {
  FeatureMap feature_id_map;
  FeatureMap feature_count_map;
  FILE *fp = fopen("test-data/input2", "r");
  Process(fp, stdout, &feature_id_map, &feature_count_map);
  fclose(fp);

  FeatureMap::const_iterator first = feature_id_map.begin();
  FeatureMap::const_iterator last = feature_id_map.end();
  for (; first != last; ++first) {
    Debug("%s -> %d [%d]\n",
          first->first.c_str(),
          first->second,
          feature_count_map[first->first]);
  }
  return 0;
}
