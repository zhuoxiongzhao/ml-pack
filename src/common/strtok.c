/* Copyright (c) Microsoft Corporation. All rights reserved. */

#include <string.h>

/* ISO/IEC 9899 7.11.5.8 strtok. DEPRECATED.
 * Split string into tokens, and return one at a time while retaining state
 * internally.
 *
 * WARNING: Only one set of state is held and this means that the
 * WARNING: function is not thread-safe nor safe for multiple uses within
 * WARNING: one thread.
 *
 * NOTE: No library may call this function.
 */

char* __cdecl strtok(char* s1, const char* delimit) {
  static char* lastToken = NULL; /* UNSAFE SHARED STATE! */
  char* tmp;

  /* Skip leading delimiters if new string. */
  if ( s1 == NULL ) {
    s1 = lastToken;
    if (s1 == NULL) {       /* End of story? */
      return NULL;
    }
  } else {
    s1 += strspn(s1, delimit);
  }

  /* Find end of segment */
  tmp = strpbrk(s1, delimit);
  if (tmp) {
    /* Found another delimiter, split string and save state. */
    *tmp = '\0';
    lastToken = tmp + 1;
  } else {
    /* Last segment, remember that. */
    lastToken = NULL;
  }

  return s1;
}
