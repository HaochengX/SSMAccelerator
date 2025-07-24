#ifndef SSM_FUNCTIONS_H
#define SSM_FUNCTIONS_H

#include <ap_fixed.h>
#include <hls_math.h>
#include <math.h>
typedef ap_fixed<16,8> DTYPE;
#define NUM 8
#define M 4
#define N 4
static const double slopes[NUM] = {
    0.3606774163013856,
    0.4070005273391741,
    0.4595267252002979,
    0.5188842792693350,
    0.5857997973719018,
    0.6610815777716942,
    0.7455437887714578,
    0.8400000000000001
};

static const double intercepts[NUM] = {
    0.8606774163013856,
    0.8220005273391741,
    0.7795267252002979,
    0.7326342792693350,
    0.6807997973719018,
    0.6234565777716942,
    0.5599062887714578,
    0.4900000000000001
};
static const DTYPE LOG2E_CONST = (DTYPE)1.4426950408889634;
DTYPE exp1(DTYPE x);
DTYPE softplus(DTYPE x);
void element_wise_mm(DTYPE A[M][N], DTYPE B[M][N], DTYPE C[M][N], int rows, int cols);
void ssm(DTYPE A[M][N], DTYPE B[M][N], DTYPE C[M][N], DTYPE D[M][N], DTYPE X[M][N], DTYPE H0[M][N], DTYPE H1[M][N], DTYPE Y[M][N], DTYPE d, DTYPE b);
#endif
