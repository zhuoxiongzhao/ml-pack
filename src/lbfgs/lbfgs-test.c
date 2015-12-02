/**
* Unconstrained Limited memory BFGS(L-BFGS).
*
* Forked from https://github.com/chokkan/liblbfgs
*
* The MIT License
*
* Copyright (c) 1990 Jorge Nocedal
* Copyright (c) 2007-2010 Naoaki Okazaki
* Copyright (c) 2014 Yafei Zhang
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*/
/**
* Unconstrained non-linear conjugate gradient.
*
* Ported from Carl Edward Rasmussen's fmincg.m.
*
* Copyright (C) 2001 and 2002 by Carl Edward Rasmussen.
* Copyright (c) 2015 Yafei Zhang
*
* Permission is granted for anyone to copy, use, or modify these
* programs and accompanying documents for purposes of research or
* education, provided this copyright notice is retained, and note is
* made of any changes that have been made.
*
* These programs and documents are distributed without any warranty,
* express or implied.  As the programs were written for research
* purposes only, they have not been tested to the degree that would be
* advisable in any important application.  All use of these programs is
* entirely at the user's own risk.
*/
#include <stdio.h>
#include "lbfgs.h"

static double evaluate(
  void* instance,
  const int n,
  const double* x,
  double* g,
  const double step
) {
  int i;
  double fx = 0.0;

  for (i = 0; i < n; i += 2) {
    double t1 = 1.0 - x[i];
    double t2 = 10.0 * (x[i + 1] - x[i] * x[i]);
    g[i + 1] = 20.0 * t2;
    g[i] = -2.0 * (x[i] * g[i + 1] + t1);
    fx += t1 * t1 + t2 * t2;
  }
  return fx;
}

#define N 100

int main(int argc, char* argv[]) {
  int i, ret = 0;
  double fx;
  double* x = vecalloc(N);
  lbfgs_parameter_t param;

  /* Initialize the variables. */
  for (i = 0; i < N; i += 2) {
    x[i] = -1.2;
    x[i + 1] = 1.0;
  }

  lbfgs_default_parameter(&param);
  param.orthantwise_c = 1.0;
  param.linesearch = LBFGS_LINESEARCH_BACKTRACKING_WOLFE;
  ret = lbfgs(N, x, &fx, evaluate, 0, 0, &param);
  vecfree(x);

  printf("L-BFGS optimization terminated with status code = %d\n", ret);
  return ret;
}
