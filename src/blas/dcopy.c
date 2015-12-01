void dcopy
(
  const int                  N,
  const double*                X,
  const int                  INCX,
  double*                      Y,
  const int                  INCY
) {
  /*
   * Purpose
   * =======
   *
   * dcopy copies the entries of an n-vector x into an n-vector y.
   *
   * Arguments
   * =========
   *
   * N       (input)                       const int
   *         On entry, N specifies the length of the vector x. N  must  be
   *         at least zero. Unchanged on exit.
   *
   * X       (input)                       const double *
   *         On entry,  X  points to the  first entry to be accessed of an
   *         incremented array of size equal to or greater than
   *            ( 1 + ( n - 1 ) * abs( INCX ) ) * sizeof(   double  ),
   *         that contains the vector x. Unchanged on exit.
   *
   * INCX    (input)                       const int
   *         On entry, INCX specifies the increment for the elements of X.
   *         INCX must not be zero. Unchanged on exit.
   *
   * Y       (input/output)                double *
   *         On entry,  Y  points to the  first entry to be accessed of an
   *         incremented array of size equal to or greater than
   *            ( 1 + ( n - 1 ) * abs( INCY ) ) * sizeof(   double  ),
   *         that contains the vector y.  On exit,  the entries of the in-
   *         cremented array  X are  copied into the entries of the incre-
   *         mented array  Y.
   *
   * INCY    (input)                       const int
   *         On entry, INCY specifies the increment for the elements of Y.
   *         INCY must not be zero. Unchanged on exit.
   *
   * ---------------------------------------------------------------------
   */
  /*
   * .. Local Variables ..
   */
  register double            x0, x1, x2, x3, x4, x5, x6, x7;
  double*                      StX;
  register int               i;
  int                        nu;
  const int                  incX2 = 2 * INCX, incY2 = 2 * INCY,
                             incX3 = 3 * INCX, incY3 = 3 * INCY,
                             incX4 = 4 * INCX, incY4 = 4 * INCY,
                             incX5 = 5 * INCX, incY5 = 5 * INCY,
                             incX6 = 6 * INCX, incY6 = 6 * INCY,
                             incX7 = 7 * INCX, incY7 = 7 * INCY,
                             incX8 = 8 * INCX, incY8 = 8 * INCY;
  /* ..
   * .. Executable Statements ..
   *
   */
  if ( N > 0 ) {
    if ( ( nu = ( N >> 3 ) << 3 ) != 0 ) {
      StX = (double*)X + nu * INCX;

      do {
        x0 = (*X);
        x4 = X[incX4];
        x1 = X[INCX ];
        x5 = X[incX5];
        x2 = X[incX2];
        x6 = X[incX6];
        x3 = X[incX3];
        x7 = X[incX7];

        *Y       = x0;
        Y[incY4] = x4;
        Y[INCY ] = x1;
        Y[incY5] = x5;
        Y[incY2] = x2;
        Y[incY6] = x6;
        Y[incY3] = x3;
        Y[incY7] = x7;

        X  += incX8;
        Y  += incY8;

      } while ( X != StX );
    }

    for ( i = N - nu; i != 0; i-- ) {
      x0  = (*X);
      *Y  = x0;

      X  += INCX;
      Y  += INCY;
    }
  }
}
