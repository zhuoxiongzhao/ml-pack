// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//

#include "hash-function.h"

/*lint -e737 all conversions in the file are OK */
/*lint -e744 'switch' statement has no 'default' */

/**
 * #define hashsize(n) ((uint32_t)1<<(n))
 * #define hashmask(n) (hashsize(n)-1)
 */
#define rot(x,k) (((x)<<(k)) | ((x)>>(32-(k))))

#define mix(a,b,c) \
  { \
    a -= c;  a ^= rot(c, 4);  c += b; \
    b -= a;  b ^= rot(a, 6);  a += c; \
    c -= b;  c ^= rot(b, 8);  b += a; \
    a -= c;  a ^= rot(c,16);  c += b; \
    b -= a;  b ^= rot(a,19);  a += c; \
    c -= b;  c ^= rot(b, 4);  b += a; \
  }

#define final(a,b,c) \
  { \
    c ^= b; c -= rot(b,14); \
    a ^= c; a -= rot(c,11); \
    b ^= a; b -= rot(a,25); \
    c ^= b; c -= rot(b,16); \
    a ^= c; a -= rot(c,4);  \
    b ^= a; b -= rot(a,14); \
    c ^= b; c -= rot(b,24); \
  }

/**
 * jenkins_hash() -- hash a variable-length key into a 32-bit value
 * k       : the key (the unaligned variable-length array of bytes)
 * length  : the length of the key, counting by bytes
 * initval : can be any 4-byte value
 * Returns a 32-bit value.  Every bit of the key affects every bit of
 * the return value.  Two keys differing by one or two bits will have
 * totally different hash values.

 * The best hash table sizes are powers of 2.  There is no need to do
 * mod a prime (mod is sooo slow!).  If you need less than 32 bits,
 * use a bitmask.  For example, if you need only 10 bits, do
 * h = (h & hashmask(10));
 * In which case, the hash table should have hashsize(10) elements.
 */
uint32_t jenkins_hash(const char* key, size_t length, uint32_t initval) {
  uint32_t a, b, c;/* internal state */
  union {
    const void* ptr;
    size_t i;
  } u;/* needed for Mac Powerbook G4 */

  if (NULL == key || 0 == length) {
    return 0;
  }

  /* Set up the internal state */
  a = b = c = 0xdeadbeef + ((uint32_t)length) + initval;

  u.ptr = key;
#ifndef WORDS_BIGENDIAN

  if ((u.i & 0x3) == 0) {
    const uint32_t* k = (const uint32_t*)key; /* read 32-bit chunks */

    /* all but last block: aligned reads and affect 32 bits of (a,b,c) */
    while (length > 12) {
      a += k[0];
      b += k[1];
      c += k[2];
      mix(a, b, c);
      length -= 12;
      k += 3;
    }

    /* handle the last (probably partial) block */
    /**
     * "k[2]&0xffffff" actually reads beyond the end of the string, but
     * then masks off the part it's not allowed to read.  Because the
     * string is aligned, the masked-off tail is in the same word as the
     * rest of the string.  Every machine with memory protection I've seen
     * does it on word boundaries, so is OK with this.  But VALGRIND will
     * still catch it and complain.  The masking trick does make the hash
     * noticably faster for short strings (like English words).
     */
    switch (length) {
    case 12:
      c += k[2];
      b += k[1];
      a += k[0];
      break;
    case 11:
      c += k[2] & 0xffffff;
      b += k[1];
      a += k[0];
      break;
    case 10:
      c += k[2] & 0xffff;
      b += k[1];
      a += k[0];
      break;
    case 9 :
      c += k[2] & 0xff;
      b += k[1];
      a += k[0];
      break;
    case 8 :
      b += k[1];
      a += k[0];
      break;
    case 7 :
      b += k[1] & 0xffffff;
      a += k[0];
      break;
    case 6 :
      b += k[1] & 0xffff;
      a += k[0];
      break;
    case 5 :
      b += k[1] & 0xff;
      a += k[0];
      break;
    case 4 :
      a += k[0];
      break;
    case 3 :
      a += k[0] & 0xffffff;
      break;
    case 2 :
      a += k[0] & 0xffff;
      break;
    case 1 :
      a += k[0] & 0xff;
      break;
    case 0 :
      return c;/* zero length strings require no mixing */
    default:
      return c;
    }

  } else if ((u.i & 0x1) == 0) {
    const uint16_t* k = (const uint16_t*)key; /* read 16-bit chunks */
    const uint8_t*  k8;

    /* all but last block: aligned reads and different mixing */
    while (length > 12) {
      a += k[0] + (((uint32_t)k[1]) << 16);
      b += k[2] + (((uint32_t)k[3]) << 16);
      c += k[4] + (((uint32_t)k[5]) << 16);
      mix(a, b, c);
      length -= 12;
      k += 6;
    }

    /* handle the last (probably partial) block */
    k8 = (const uint8_t*)k;

    switch (length) {
    case 12:
      c += k[4] + (((uint32_t)k[5]) << 16);
      b += k[2] + (((uint32_t)k[3]) << 16);
      a += k[0] + (((uint32_t)k[1]) << 16);
      break;
    case 11:
      c += ((uint32_t)k8[10]) << 16;
    /*lint -fallthrough */
    case 10:
      c += k[4];
      b += k[2] + (((uint32_t)k[3]) << 16);
      a += k[0] + (((uint32_t)k[1]) << 16);
      break;
    case 9 :
      c += k8[8];
    /*lint -fallthrough */
    case 8 :
      b += k[2] + (((uint32_t)k[3]) << 16);
      a += k[0] + (((uint32_t)k[1]) << 16);
      break;
    case 7 :
      b += ((uint32_t)k8[6]) << 16;
    /*lint -fallthrough */
    case 6 :
      b += k[2];
      a += k[0] + (((uint32_t)k[1]) << 16);
      break;
    case 5 :
      b += k8[4];
    /*lint -fallthrough */
    case 4 :
      a += k[0] + (((uint32_t)k[1]) << 16);
      break;
    case 3 :
      a += ((uint32_t)k8[2]) << 16;
    /*lint -fallthrough */
    case 2 :
      a += k[0];
      break;
    case 1 :
      a += k8[0];
      break;
    case 0 :
      return c;/* zero length requires no mixing */
    default:
      return c;
    }

  } else { /* need to read the key one byte at a time */
#endif /* little endian */
    const uint8_t* k = (const uint8_t*)key;

    /* all but the last block: affect some 32 bits of (a,b,c) */
    while (length > 12) {
      a += k[0];
      a += ((uint32_t)k[1]) << 8;
      a += ((uint32_t)k[2]) << 16;
      a += ((uint32_t)k[3]) << 24;
      b += k[4];
      b += ((uint32_t)k[5]) << 8;
      b += ((uint32_t)k[6]) << 16;
      b += ((uint32_t)k[7]) << 24;
      c += k[8];
      c += ((uint32_t)k[9]) << 8;
      c += ((uint32_t)k[10]) << 16;
      c += ((uint32_t)k[11]) << 24;
      mix(a, b, c);
      length -= 12;
      k += 12;
    }

    /* last block: affect all 32 bits of (c) */
    switch (length) { /* all the case statements fall through */
    case 12:
      c += ((uint32_t)k[11]) << 24;
    /*lint -fallthrough */
    case 11:
      c += ((uint32_t)k[10]) << 16;
    /*lint -fallthrough */
    case 10:
      c += ((uint32_t)k[9]) << 8;
    /*lint -fallthrough */
    case 9 :
      c += k[8];
    /*lint -fallthrough */
    case 8 :
      b += ((uint32_t)k[7]) << 24;
    /*lint -fallthrough */
    case 7 :
      b += ((uint32_t)k[6]) << 16;
    /*lint -fallthrough */
    case 6 :
      b += ((uint32_t)k[5]) << 8;
    /*lint -fallthrough */
    case 5 :
      b += k[4];
    /*lint -fallthrough */
    case 4 :
      a += ((uint32_t)k[3]) << 24;
    /*lint -fallthrough */
    case 3 :
      a += ((uint32_t)k[2]) << 16;
    /*lint -fallthrough */
    case 2 :
      a += ((uint32_t)k[1]) << 8;
    /*lint -fallthrough */
    case 1 :
      a += k[0];
      break;
    case 0 :
      return c;
    default :
      return c;
    }

#ifndef  WORDS_BIGENDIAN
  }

#endif

  final(a, b, c);
  return c;
}


uint32_t jenkins_hash2(const char* key, size_t length) {
  return jenkins_hash(key, length, 0);
}


#ifdef get16bits
# undef get16bits
#endif

#if (defined(__GNUC__) && defined(__i386__))
# define get16bits(d) (*((const uint16_t *) (d)))
#endif

#if !defined (get16bits)
# define get16bits(d) ((((uint32_t)(((const uint8_t *)(d))[1])) << 8)\
                       +(uint32_t)(((const uint8_t *)(d))[0]) )
#endif


uint32_t hsieh_hash(const char* key, size_t length) {
  uint32_t hash = 0, tmp;
  int rem;
  const char* data = (const char*)key;

  if (NULL == key || 0 == length) {
    return 0;
  }

  rem = length & 3;
  length >>= 2;

  /* Main loop */
  for (; length > 0; length--) {
    hash  += get16bits(data);
    tmp    = (get16bits(data + 2) << 11) ^ hash;
    hash   = (hash << 16) ^ tmp;
    data  += 2 * sizeof(uint16_t);
    hash  += hash >> 11;
  }

  /* Handle end cases */
  switch (rem) {
  case 3:
    hash += get16bits(data);
    hash ^= hash << 16;
    hash ^= (unsigned char)data[sizeof(uint16_t)] << 18;
    hash += hash >> 11;
    break;
  case 2:
    hash += get16bits(data);
    hash ^= hash << 11;
    hash += hash >> 17;
    break;
  case 1:
    hash += (unsigned char) * data;
    hash ^= hash << 10;
    hash += hash >> 1;
  }

  /* Force "avalanching" of final 127 bits */
  hash ^= hash << 3;
  hash += hash >> 5;
  hash ^= hash << 4;
  hash += hash >> 17;
  hash ^= hash << 25;
  hash += hash >> 6;

  return hash;
}


uint32_t murmur_hash(const char* key, size_t length) {
  /**
   * 'm' and 'r' are mixing constants generated offline.  They're not
   * really 'magic', they just happen to work well.
   */

  const uint32_t m = 0x5bd1e995;
  const uint32_t seed = (0xdeadbeef * length);
  const int r = 24;


  /* Initialize the hash to a 'random' value */

  uint32_t h = seed ^ length;

  /* Mix 4 bytes at a time into the hash */

  const unsigned char* data = (const unsigned char*)key;

  if (NULL == key || 0 == length) {
    return 0;
  }

  while (length >= 4) {
    uint32_t k = /*lint -e(826) safe */*(uint32_t*)data;

    k *= m;
    k ^= k >> r;
    k *= m;

    h *= m;
    h ^= k;

    data += 4;
    length -= 4;
  }

  /* Handle the last few bytes of the input array */

  switch (length) {
  case 3:
    h ^= data[2] << 16;
  /*lint -fallthrough */
  case 2:
    h ^= data[1] << 8;
  /*lint -fallthrough */
  case 1:
    h ^= data[0];
    h *= m;
  /*lint -fallthrough */
  default:
    break;
  };

  /**
   * Do a few final mixes of the hash to ensure the last few bytes are
   * well-incorporated.
   */

  h ^= h >> 13;

  h *= m;

  h ^= h >> 15;

  return h;
}


uint32_t bkdr_hash(const char* key, size_t length) {
  static const uint32_t kHashMaxValue = 0xffffffff;

  uint32_t seed = 131;/* 31 131 1313 13131 131313 etc.. */
  uint32_t hash = 0;
  size_t i;
  const unsigned char* data = (const unsigned char*)key;

  if (NULL == key || 0 == length) {
    return 0;
  }

  for (i = 0; i < length; i++) {
    hash = hash * seed + (*data++);
  }

  return hash & kHashMaxValue;
}


uint32_t ap_hash(const char* key, size_t length) {
  uint32_t hash = 0xAAAAAAAA;
  size_t i;
  const unsigned char* data = (const unsigned char*)key;

  if (NULL == key || 0 == length) {
    return 0;
  }

  for (i = 0; i < length; i++) {
    hash ^= ((i & 1) == 0) ? (  (hash <<  7) ^ data[i] * (hash >> 3)) :
            (~((hash << 11) + (data[i] ^ (hash >> 5))));
  }

  return hash;
}


uint32_t djb_hash(const char* key, size_t length) {
  uint32_t hash = 5381;
  size_t i;
  const unsigned char* data = (const unsigned char*)key;

  if (NULL == key || 0 == length) {
    return 0;
  }

  for (i = 0; i < length; i++) {
    hash = ((hash << 5) + hash) + data[i];
  }

  return hash;
}


uint32_t fnv_hash_32(const char* key, size_t length) {
  static const uint32_t kHashMaxValue = 0xffffffff;
  static const uint64_t kFnv64Prime = UINT64_C(1099511628211);
  static const uint64_t kFnv64Offset = UINT64_C(14695981039346656037);

  size_t i;
  uint64_t hash = kFnv64Offset;
  const unsigned char* data = (const unsigned char*)key;

  if (NULL == key || 0 == length) {
    return 0;
  }

  for (i = 0; i < length; i++) {
    hash ^= (uint64_t)data[i];
    hash *= kFnv64Prime;
  }

  return hash & kHashMaxValue;
}


uint64_t fnv_hash_64(const char* key, size_t length) {
  static const uint32_t kHashMaxValue = 0xffffffff;
  static const uint64_t kFnv64Prime = UINT64_C(1099511628211);
  static const uint64_t kFnv64Offset = UINT64_C(14695981039346656037);

  size_t i;
  uint64_t hash = kFnv64Offset;
  const unsigned char* data = (const unsigned char*)key;

  if (NULL == key || 0 == length) {
    return 0;
  }

  for (i = 0; i < length; i++) {
    hash ^= (uint64_t)data[i];
    hash *= kFnv64Prime;
  }

  return hash & kHashMaxValue;
}


uint32_t sdbm_hash(const char* key, size_t length) {
  uint32_t hash = 0;
  size_t i;
  const unsigned char* data = (const unsigned char*)key;

  if (NULL == key || 0 == length) {
    return 0;
  }

  for (i = 0; i < length; i++) {
    hash = data[i] + (hash << 6) + (hash << 16) - hash;
  }

  return hash;
}


uint32_t rs_hash(const char* key, size_t length) {
  uint32_t a = 378551;
  uint32_t b = 63689;
  uint32_t hash = 0;
  size_t i;
  const char* data = (const char*)key;

  if (NULL == key || 0 == length) {
    return 0;
  }

  for (i = 0; i < length; i++) {
    hash = hash * a + (uint8_t)data[i];
    a *= b;
  }

  return (hash & 0x7fffffff);
}


/**
 * Generic hash function (a popular one from Bernstein)
 * Inspired by redis
 */
uint32_t time33_hash_32(const char* key, size_t length) {
  uint32_t hash = 5381;/* magic number */
  const char* str = (const char*)key;

  if (NULL == key || 0 == length) {
    return 0;
  }

  for (; length >= 8; length -= 8) {
    /* expand loop */
    hash = ((hash << 5) + hash) + *str++;
    hash = ((hash << 5) + hash) + *str++;
    hash = ((hash << 5) + hash) + *str++;
    hash = ((hash << 5) + hash) + *str++;
    hash = ((hash << 5) + hash) + *str++;
    hash = ((hash << 5) + hash) + *str++;
    hash = ((hash << 5) + hash) + *str++;
    hash = ((hash << 5) + hash) + *str++;
  }

  switch (length) {
  case 7:
    hash = ((hash << 5) + hash) + *str++; /*lint -fallthrough */
  case 6:
    hash = ((hash << 5) + hash) + *str++; /*lint -fallthrough */
  case 5:
    hash = ((hash << 5) + hash) + *str++; /*lint -fallthrough */
  case 4:
    hash = ((hash << 5) + hash) + *str++; /*lint -fallthrough */
  case 3:
    hash = ((hash << 5) + hash) + *str++; /*lint -fallthrough */
  case 2:
    hash = ((hash << 5) + hash) + *str++; /*lint -fallthrough */
  case 1:
    hash = ((hash << 5) + hash) + *str++;
    break;
  case 0:
    break;
  }

  return hash;
}


uint64_t time33_hash_64(const char* key, size_t length) {
  uint64_t hash = UINT64_C(5381);/* magic number */
  const char* str = (const char*)key;

  if (NULL == key || 0 == length) {
    return 0;
  }

  for (; length >= 8; length -= 8) {
    /* expand loop */
    hash = ((hash << 5) + hash) + *str++;
    hash = ((hash << 5) + hash) + *str++;
    hash = ((hash << 5) + hash) + *str++;
    hash = ((hash << 5) + hash) + *str++;
    hash = ((hash << 5) + hash) + *str++;
    hash = ((hash << 5) + hash) + *str++;
    hash = ((hash << 5) + hash) + *str++;
    hash = ((hash << 5) + hash) + *str++;
  }

  switch (length) {
  case 7:
    hash = ((hash << 5) + hash) + *str++; /*lint -fallthrough */
  case 6:
    hash = ((hash << 5) + hash) + *str++; /*lint -fallthrough */
  case 5:
    hash = ((hash << 5) + hash) + *str++; /*lint -fallthrough */
  case 4:
    hash = ((hash << 5) + hash) + *str++; /*lint -fallthrough */
  case 3:
    hash = ((hash << 5) + hash) + *str++; /*lint -fallthrough */
  case 2:
    hash = ((hash << 5) + hash) + *str++; /*lint -fallthrough */
  case 1:
    hash = ((hash << 5) + hash) + *str++;
    break;
  case 0:
    break;
  }

  return hash;
}
