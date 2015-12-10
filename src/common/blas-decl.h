// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// BLAS function declaration
//

#ifndef SRC_COMMON_BLAS_DECL_H_
#define SRC_COMMON_BLAS_DECL_H_

#if defined __cplusplus
# define EXTERN extern "C"
#else
# define EXTERN extern
#endif

EXTERN void dcopy
(
  const int                  N,
  const double*              X,
  const int                  INCX,
  double*                    Y,
  const int                  INCY
);

EXTERN void dscal
(
  const int                  N,
  const double               ALPHA,
  double*                    X,
  const int                  INCX
);

EXTERN void daxpy
(
  const int                  N,
  const double               ALPHA,
  const double*              X,
  const int                  INCX,
  double*                    Y,
  const int                  INCY
);

EXTERN double ddot
(
  const int                  N,
  const double*              X,
  const int                  INCX,
  const double*              Y,
  const int                  INCY
);

EXTERN void dswap
(
  const int                  N,
  double*                    X,
  const int                  INCX,
  double*                    Y,
  const int                  INCY
);

#endif  // SRC_COMMON_BLAS_DECL_H_

