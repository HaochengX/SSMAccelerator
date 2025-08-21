#include "lightmamba.h"
void silu(DTYPE in[N], DTYPE out[N]){
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=in  type=block factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=out type=block factor=4 dim=1
	for(int j=0; j<N; j++){
#pragma HLS UNROLL
		out[j] = in[j] * (DTYPE)(1.0f / (1.0f + hls::expf(-(float)in[j])));
	}
}

DTYPE softplus(DTYPE x){
	return (DTYPE)(hls::logf(1.0f + hls::expf((float)x)));
}

void exp1(DTYPE in[N], DTYPE out[N]){
#pragma HLS INLINE off
	for(int j=0; j<N; j++){
#pragma HLS UNROLL
		out[j] = (DTYPE)hls::expf((float)in[j]);
	}
}

void conv1d(DTYPE input_X[INPUT_DIM], DTYPE kernel[K], DTYPE Y[N]) {
#pragma HLS ARRAY_PARTITION variable=kernel type=block dim=1 factor=2
	DTYPE window[K];
#pragma HLS ARRAY_PARTITION variable=window type=block dim=1 factor=2

	for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
		for (int j = K-1; j > 0; j--) {
			window[j] = window[j-1];
		}
		window[0] = input_X[i + K - 1];

		DTYPE sum = 0;
		for (int j = 0; j < K; j++) {
			sum += kernel[j] * window[j];
		}
		Y[i] = sum;
	}
}
template <int size>
void EMU(DTYPE A[size], DTYPE B[size], DTYPE out[size]){
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=A   type=block factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=B   type=block factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=out type=block factor=4 dim=1
	for(int j=0; j<size; j++){
#pragma HLS UNROLL
		out[j]=A[j]*B[j];
	}
}


template <int size>
void EAU(DTYPE A[size], DTYPE B[size], DTYPE out[size]){
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=A type=block factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=B type=block factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=out type=block factor=4 dim=1
	for(int j=0; j<size; j++){
#pragma HLS UNROLL
		out[j]=A[j]+B[j];
	}
}

void SSMU(
DTYPE kernel[K], DTYPE A[N], DTYPE B[N], DTYPE C[N], DTYPE D[N],
DTYPE X[N], DTYPE Z[N],
DTYPE H0[M][N], DTYPE H1[M][N],
DTYPE delta[N], DTYPE bias[N], DTYPE out[N]){

#pragma HLS INTERFACE m_axi port=X offset=slave bundle=data_in
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=data_out
#pragma HLS INTERFACE m_axi port=H0 offset=slave bundle=data_Hin
#pragma HLS INTERFACE m_axi port=H1 offset=slave bundle=data_Hout
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=data0
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=data1
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=data2
#pragma HLS INTERFACE m_axi port=D offset=slave bundle=data3
#pragma HLS INTERFACE m_axi port=Z offset=slave bundle=data4
#pragma HLS INTERFACE m_axi port=delta offset=slave bundle=data5
#pragma HLS INTERFACE m_axi port=bias offset=slave bundle=data6
#pragma HLS INTERFACE m_axi port=kernel offset=slave bundle=data7
#pragma HLS INTERFACE s_axilite register port=return bundle=ctrl

#pragma HLS ALLOCATION function instances=silu limit=1

    DTYPE A_buffer[N];
    DTYPE B_buffer[N];
    DTYPE C_buffer[N];
    DTYPE D_buffer[N];
    DTYPE Z_buffer[N];
    DTYPE X_buffer[N];
    DTYPE delta_buffer[N];
    DTYPE bias_buffer[N];
    DTYPE kernel_buffer[K];
    DTYPE out_buffer[N];

#pragma HLS ARRAY_PARTITION variable=A_buffer type=block factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=B_buffer type=block factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=C_buffer type=block factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=D_buffer type=block factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=Z_buffer type=block factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=X_buffer type=block factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=bias_buffer type=block factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=delta_buffer type=block factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=kernel_buffer type=block factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=out_buffer type=block factor=4 dim=1

    for(int i=0; i<N; i++){
#pragma HLS PIPELINE II=1
        A_buffer[i] = A[i];
        B_buffer[i] = B[i];
        C_buffer[i] = C[i];
        D_buffer[i] = D[i];
        Z_buffer[i] = Z[i];
        X_buffer[i] = X[i];
        bias_buffer[i] = bias[i];
        delta_buffer[i] = delta[i];
    }
    for(int i=0; i<K; i++){
#pragma HLS PIPELINE II=1
        kernel_buffer[i] = kernel[i];
    }

    DTYPE dd[N], dA[N], dB[N], dC[N], dX[N], dZ[N], ddA[N], ddX[N], ddB[N], yy1[N], accu_sum[N];
    EAU<N>(delta_buffer, bias_buffer, dd);
    for(int j=0; j<N; j++){
#pragma HLS PIPELINE II=1
	    dd[j]=softplus(dd[j]);
    }
    conv1d(X_buffer, kernel_buffer, dX);
    silu(B_buffer, dB);
    silu(C_buffer, dC);
    silu(dX, ddX);
    silu(Z_buffer, dZ);
    EMU<N>(dB, dd, ddB);
    EMU<N>(A_buffer, dd, dA);
    EMU<N>(ddX, D_buffer, yy1);
    exp1(dA, ddA);
	for (int i = 0; i < N; i++) {
		accu_sum[i] = 0;
	}
	for (int i_tile = 0; i_tile < M/np; i_tile++) {
		for (int j_tile = 0; j_tile < N/pp; j_tile++) {
			DTYPE ddA_tile[pp];
			DTYPE ddB_tile[pp];
			DTYPE ddX_tile[pp];
			DTYPE H0_tile[np][pp];
			DTYPE dC_tile[pp];
			for (int k = 0; k < pp; k++) {
				ddA_tile[k] = ddA[j_tile * pp + k];
				ddB_tile[k] = ddB[j_tile * pp + k];
				ddX_tile[k] = ddX[j_tile * pp + k];
				dC_tile[k] = dC[j_tile * pp + k];
			}
			for (int i = 0; i < np; i++) {
#pragma HLS UNROLL
				for (int k = 0; k < pp; k++) {
					H0_tile[i][k] = H0[i_tile * np + i][j_tile * pp + k];
				}
			}
			DTYPE hh0_local[np][pp];
			DTYPE hh1_local[np][pp];
			DTYPE dH_local[np][pp];
			DTYPE H1_write_back[np][pp];
			DTYPE accu_sum_tile[pp];
			for (int i = 0; i < np; i++) {
#pragma HLS PIPELINE II=1
			EMU<pp>(ddA_tile, H0_tile[i], hh0_local[i]);
			EMU<pp>(ddB_tile, ddX_tile, hh1_local[i]);
			EAU<pp>(hh0_local[i], hh1_local[i], H1_write_back[i]);
			EMU<pp>(dC_tile, H1_write_back[i], dH_local[i]);
			}
			for (int kk=0; kk<pp; kk++) accu_sum_tile[kk]=0;
			for (int ii=0; ii<np; ii++)
#pragma HLS UNROLL
			  for (int kk=0; kk<pp; kk++)
			    accu_sum_tile[kk] += dH_local[ii][kk];

			for (int i=0; i<pp; i++) {
				accu_sum[j_tile*pp + i] += accu_sum_tile[i];
			}
			for (int i = 0; i < np; i++) {
#pragma HLS UNROLL
				for (int k = 0; k < pp; k++) {
					H1[i_tile * np + i][j_tile * pp + k] = H1_write_back[i][k];
				}
			}
		}
	}
	DTYPE Y[N];
	EAU<N>(accu_sum, yy1, Y);
	EMU<N>(Y, dZ, out_buffer);
	for(int i=0; i<N;i++){
#pragma HLS UNROLL
		out[i]=out_buffer[i];
	}
}
