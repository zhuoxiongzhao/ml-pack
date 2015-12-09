// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//

#include <string.h>

#include "hash/hash-entry.h"
#include "hash/city.h"
#include "hash/hash-function.h"

typedef uint32_t (* Hash32Func)(const char* key, size_t length);

Hash32Func hash_func[] = {
  &jenkins_hash2,
  &hsieh_hash,
  &murmur_hash,
  &bkdr_hash,
  &ap_hash,
  &djb_hash,
  &fnv_hash_32,
  &sdbm_hash,
  &rs_hash,
  &time33_hash_32,
  &CityHash32
};

// Please select an ideal hash function.
Hash32Func the_hash_func = hash_func[10];

int HashString(const std::string& s) {
  return (int)the_hash_func(s.c_str(), s.size());
}
