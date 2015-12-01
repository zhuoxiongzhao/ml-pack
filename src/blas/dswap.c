void dswap
(
  const int                  N,
  double*                      X,
  const int                  INCX,
  double*                      Y,
  const int                  INCY
) {
  /*
   * Purpose
   * =======
   *
   * dswap swaps the entries of two n-vectors x and y.
   *
   * Arguments
   * =========
   *
   * N       (input)                       const int
   *         On entry, N specifies the length of the vector x. N  must  be
   *         at least zero. Unchanged on exit.
   *
   * X       (input/output)                double *
   *         On entry,  X  points to the  first entry to be accessed of an
   *         incremented array of size equal to or greater than
   *            ( 1 + ( n - 1 ) * abs( INCX ) ) * sizeof(   double  ),
   *         that contains the vector x.  On exit,  the entries of the in-
   *         cremented array  X are swapped with the entries of the incre-
   *         mented array  Y.
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
   *         cremented array  Y are swapped with the entries of the incre-
   *         mented array  X.
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
  register double            x0, x1, x2, x3, y0, y1, y2, y3;
  double*                      StX;
  register int               i;
  int                        nu;
  const int                  incX2 = 2 * INCX, incY2 = 2 * INCY,
                             incX3 = 3 * INCX, incY3 = 3 * INCY,
                             incX4 = 4 * INCX, incY4 = 4 * INCY;
  /* ..
   * .. Executable Statements ..
   *
   */
  if ( N > 0 ) {
    if ( ( nu = ( N >> 2 ) << 2 ) != 0 ) {
      StX = (double*)X + nu * INCX;

      do {
        x0 = (*X);
        y0 = (*Y);
        x1 = X[INCX ];
        y1 = Y[INCY ];
        x2 = X[incX2];
        y2 = Y[incY2];
        x3 = X[incX3];
        y3 = Y[incY3];

        *Y        = x0;
        *X        = y0;
        Y[INCY ]  = x1;
        X[INCX ]  = y1;
        Y[incY2]  = x2;
        X[incX2]  = y2;
        Y[incY3]  = x3;
        X[incX3]  = y3;

        X += incX4;
        Y += incY4;

      } while ( X != StX );
    }

    for ( i = N - nu; i != 0; i-- ) {
      x0  = (*X);
      y0  = (*Y);

      *Y  = x0;
      *X  = y0;

      X  += INCX;
      Y  += INCY;
    }
  }
}
