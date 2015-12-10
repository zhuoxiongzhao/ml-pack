double ddot
(
  const int                  N,
  const double*                X,
  const int                  INCX,
  const double*                Y,
  const int                  INCY
) {
  /*
   * Purpose
   * =======
   *
   * ddot returns the dot product x^T * y of two n-vectors x and y.
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
   * Y       (input)                       const double *
   *         On entry,  Y  points to the  first entry to be accessed of an
   *         incremented array of size equal to or greater than
   *            ( 1 + ( n - 1 ) * abs( INCY ) ) * sizeof(   double  ),
   *         that contains the vector y. Unchanged on exit.
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
  register double            dot = 0.0, x0, x1, x2, x3,
                             y0, y1, y2, y3;
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
        dot += x0 * y0;
        dot += x1 * y1;
        dot += x2 * y2;
        dot += x3 * y3;
        X  += incX4;
        Y  += incY4;
      } while ( X != StX );
    }

    for ( i = N - nu; i != 0; i-- ) {
      x0 = (*X);
      y0 = (*Y);
      dot += x0 * y0;
      X += INCX;
      Y += INCY;
    }
  }
  return ( dot );
}
