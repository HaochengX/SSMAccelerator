#include "ssm_accelerator.h"
DTYPE exp1(DTYPE x){
#pragma HLS INLINE
	DTYPE v;
	int u;
	DTYPE y;
	DTYPE u2;
	DTYPE v2;
	y=LOG2E_CONST*x;
	u=(int)hls::floor(y);
	v=y-(DTYPE)u;
	if (v==0.0) {v2 = (DTYPE)1.0;}
	else{
		int index=(int)hls::floor(((v+(DTYPE)1.0) * (DTYPE)NUM));
		if (index < 0) index = 0;
		if (index >= NUM) index = NUM - 1;
	    v2 = (DTYPE)slopes[index]*v+(DTYPE)intercepts[index];
	}
	if (u == 0) {u2=(DTYPE)1.0;}
	else {u2=(DTYPE)1.0 >> (-u);}
	return u2*v2;
}

DTYPE softplus(DTYPE x){
#pragma HLS INLINE
	if(x<=0){
		return exp1(x);
	}else {
		return x+exp1(-x);
	}
}

void element_wise_mm(DTYPE A[M][N], DTYPE B[M][N], DTYPE C[M][N], int rows, int cols){
#pragma HLS ARRAY_PARTITION variable=A type=complete dim=0
#pragma HLS ARRAY_PARTITION variable=B type=complete dim=0
#pragma HLS ARRAY_PARTITION variable=C type=complete dim=0
	for(int i=0; i<rows; i++){
#pragma HLS PIPELINE II=1
		for(int j=0; j<cols; j++){
			C[i][j]=(DTYPE)A[i][j]*B[i][j];
		}
	}
}

void ssm(DTYPE A[M][N], DTYPE B[M][N], DTYPE C[M][N], DTYPE D[M][N], DTYPE X[M][N], DTYPE H0[M][N], DTYPE H1[M][N], DTYPE Y[M][N], DTYPE d, DTYPE b){
#pragma HLS DATAFLOW
	DTYPE dd[M][N];
	DTYPE dA[M][N];
	DTYPE dB[M][N];
    DTYPE hh0[M][N];
    DTYPE hh1[M][N];
    DTYPE yy0[M][N];
    DTYPE yy1[M][N];
#pragma HLS ARRAY_PARTITION variable=dd complete dim=0
#pragma HLS ARRAY_PARTITION variable=dA complete dim=0
#pragma HLS ARRAY_PARTITION variable=dB complete dim=0
#pragma HLS ARRAY_PARTITION variable=hh0 complete dim=0
#pragma HLS ARRAY_PARTITION variable=hh1 complete dim=0
#pragma HLS ARRAY_PARTITION variable=yy0 complete dim=0
#pragma HLS ARRAY_PARTITION variable=yy1 complete dim=0

	for(int i=0; i<M; i++){
		for(int j=0; j<N; j++){
			dd[i][j]=softplus(d+b);
		}
	}
	element_wise_mm(A, dd, dA, M, N);
	element_wise_mm(B, dd, dB, M, N);
	for(int i=0; i<M; i++){
#pragma HLS PIPELINE II=1
		for(int j=0; j<N; j++){
			dA[i][j]=expf(dA[i][j]);
		}
	}
	element_wise_mm(dA, H0, hh0, M, N);
	element_wise_mm(dB, X, hh1, M, N);
	for(int i=0; i<M; i++){
#pragma HLS PIPELINE II=1
			for(int j=0; j<N; j++){
				H1[i][j]=hh0[i][j]+hh1[i][j];
			}
		}
	element_wise_mm(D, X, yy0, M, N);
	element_wise_mm(C, H0, yy1, M, N);
	for(int i=0; i<M; i++){
#pragma HLS PIPELINE II=1
		for(int j=0; j<N; j++){
			Y[i][j]=yy0[i][j]+yy1[i][j];
		}
	}
}
