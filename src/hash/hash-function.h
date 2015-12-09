// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// some hash functions
//

#ifndef HASH_FUNCTION_H
#define HASH_FUNCTION_H

#include <stddef.h>
#include <stdint.h>

#if defined __cplusplus
extern "C" {
#endif

  uint32_t jenkins_hash(const char* key, size_t length, uint32_t initval);
  uint32_t jenkins_hash2(const char* key, size_t length);
  uint32_t hsieh_hash(const char* key, size_t length);
  uint32_t murmur_hash(const char* key, size_t length);
  uint32_t bkdr_hash(const char* key, size_t length);
  uint32_t ap_hash(const char* key, size_t length);
  uint32_t djb_hash(const char* key, size_t length);
  uint32_t fnv_hash_32(const char* key, size_t length);
  uint64_t fnv_hash_64(const char* key, size_t length);
  uint32_t sdbm_hash(const char* key, size_t length);
  uint32_t rs_hash(const char* key, size_t length);
  uint32_t time33_hash_32(const char* key, size_t length);
  uint64_t time33_hash_64(const char* key, size_t length);

#if defined __cplusplus
}
#endif

#endif/* HASH_FUNCTION_H */
