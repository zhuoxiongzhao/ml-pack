// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// BLAS function declaration
//

#ifndef SRC_BLAS_BLAS_DECL_H_
#define SRC_BLAS_BLAS_DECL_H_

extern "C" void dcopy
(
  const int                  N,
  const double*              X,
  const int                  INCX,
  double*                    Y,
  const int                  INCY
);

extern "C" void dscal
(
  const int                  N,
  const double               ALPHA,
  double*                    X,
  const int                  INCX
);

extern "C" void daxpy
(
  const int                  N,
  const double               ALPHA,
  const double*              X,
  const int                  INCX,
  double*                    Y,
  const int                  INCY
);

extern "C" double ddot
(
  const int                  N,
  const double*              X,
  const int                  INCX,
  const double*              Y,
  const int                  INCY
);

extern "C" void dswap
(
  const int                  N,
  double*                    X,
  const int                  INCX,
  double*                    Y,
  const int                  INCY
);

#endif  // SRC_BLAS_BLAS_DECL_H_

