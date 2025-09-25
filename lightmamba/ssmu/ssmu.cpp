#include "ssmu.h"
//silu
void silu(DTYPE_VEC in[VEC_N], DTYPE_VEC out[VEC_N]) {
    silu_vec_loop: for(int j = 0; j < VEC_N; j++) {
        #pragma HLS PIPELINE II=1
        DTYPE_VEC tmp;
        for(int k = 0; k < VEC_FACTOR; k++) {
            #pragma HLS UNROLL
            DTYPE val = in[j][k];
            DTYPE exp_val = hls::exp(-val);
            DTYPE sigmoid = (DTYPE)1.0 / ((DTYPE)1.0 + exp_val);
            tmp[k] = val * sigmoid;
        }
        out[j] = tmp;
    }
}

//softplus
void softplus(DTYPE_VEC in[VEC_N], DTYPE_VEC out[VEC_N]) {
    softplus_loop: for(int j = 0; j < VEC_N; j++) {
        #pragma HLS PIPELINE II=1
        DTYPE_VEC tmp;
        for(int k = 0; k < VEC_FACTOR; k++) {
            #pragma HLS UNROLL
            tmp[k] = (DTYPE)(hls::log(1 + hls::exp(in[j][k])));
        }
        out[j] = tmp;
    }
}


//exp
void exp1(DTYPE_VEC in[VEC_N], DTYPE_VEC out[VEC_N]) {
    exp1_loop: for(int j = 0; j < VEC_N; j++) {
        #pragma HLS PIPELINE II=1
        DTYPE_VEC tmp;
        for(int k = 0; k < VEC_FACTOR; k++) {
            #pragma HLS UNROLL
            tmp[k] = (DTYPE)hls::exp(in[j][k]);
        }
        out[j] = tmp;
    }
}

//1 dimension conv layer
void conv1d_vec(DTYPE_VEC input_X[VEC_N], DTYPE kernel[K], DTYPE_VEC Y[VEC_N]) {
    DTYPE input_scalar[INPUT_DIM];
    unpack_input: for(int i = 0; i < VEC_N; i++) {
        for(int k = 0; k < VEC_FACTOR; k++) {
            #pragma HLS PIPELINE II=1
            int idx = i * VEC_FACTOR + k;
            if(idx < INPUT_DIM) {
                input_scalar[idx] = input_X[i][k];
            }
        }
    }
    DTYPE Y_scalar[N];
    conv_outer: for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        DTYPE sum = 0;
        conv_inner: for (int j = 0; j < K; j++) {
            #pragma HLS UNROLL
            sum += kernel[j] * input_scalar[i+j];
        }
        Y_scalar[i] = sum;
    }
    pack_output: for(int i = 0; i < VEC_N; i++) {
        #pragma HLS PIPELINE II=1
        DTYPE_VEC tmp;
        for(int k = 0; k < VEC_FACTOR; k++) {
            #pragma HLS UNROLL
            int idx = i * VEC_FACTOR + k;
            if(idx < N) {
                tmp[k] = Y_scalar[idx];
            } else {
                tmp[k] = 0;
            }
        }
        Y[i] = tmp;
    }
}

//Element-wise Multiplication Unit
void EMU(DTYPE_VEC A[VEC_N], DTYPE_VEC B[VEC_N], DTYPE_VEC out[VEC_N]) {
    EMU_vec_loop: for(int j = 0; j < VEC_N; j++) {
        out[j] = A[j] * B[j];
    }
}

//Element-wise addition Unit
void EAU(DTYPE_VEC A[VEC_N], DTYPE_VEC B[VEC_N], DTYPE_VEC out[VEC_N]) {
    EAU_vec_loop: for(int j = 0; j < VEC_N; j++) {
        out[j] = A[j] + B[j];
    }
}

//accumulator
void accumulate(DTYPE_VEC accumulator[VEC_N], DTYPE_VEC addend[VEC_N]) {
    accumulate_loop: for(int j = 0; j < VEC_N; j++) {
        accumulator[j] = accumulator[j] + addend[j];
    }
}

//packed transfer
//vector
//stream

//top function
void SSMU(
//DTYPE_BLOCK16 weight_pack_16[N], //16 * 8bit = 128bit
    DTYPE kernel[K],
    DTYPE_VEC A[VEC_N], DTYPE_VEC B[VEC_N], DTYPE_VEC C[VEC_N], DTYPE_VEC D[VEC_N],
    DTYPE_VEC X[VEC_N], DTYPE_VEC Z[VEC_N],
    DTYPE_VEC H0[M][VEC_N], DTYPE_VEC H1[M][VEC_N],
    DTYPE_VEC delta[VEC_N], DTYPE_VEC bias[VEC_N],
    DTYPE_VEC out[VEC_N]){

#pragma HLS INTERFACE m_axi port=X offset=slave bundle=data_in depth=VEC_N
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=data_out depth=VEC_N
#pragma HLS INTERFACE m_axi port=H0 offset=slave bundle=data_H depth=M*VEC_N
#pragma HLS INTERFACE m_axi port=H1 offset=slave bundle=data_H depth=M*VEC_N
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=data_W depth=VEC_N
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=data_W depth=VEC_N
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=data_W depth=VEC_N
#pragma HLS INTERFACE m_axi port=D offset=slave bundle=data_W depth=VEC_N
#pragma HLS INTERFACE m_axi port=Z offset=slave bundle=data_W depth=VEC_N
#pragma HLS INTERFACE m_axi port=delta offset=slave bundle=data_W depth=VEC_N
#pragma HLS INTERFACE m_axi port=bias offset=slave bundle=data_W depth=VEC_N
#pragma HLS INTERFACE m_axi port=kernel offset=slave bundle=data_W depth=K
#pragma HLS INTERFACE s_axilite register port=return bundle=ctrl

    hls::stream<DTYPE_VEC> stream_ddA;
    hls::stream<DTYPE_VEC> stream_ddX;
    hls::stream<DTYPE_VEC> stream_ddB;
    hls::stream<DTYPE_VEC> stream_dC;
    hls::stream<DTYPE_VEC> stream_H0[M];
    hls::stream<DTYPE_VEC> stream_H1[M];
    hls::stream<DTYPE_VEC> stream_dH[M];
    hls::stream<DTYPE_VEC> stream_yy0;
    
    #pragma HLS STREAM variable=stream_ddA depth=64
    #pragma HLS STREAM variable=stream_ddX depth=64
    #pragma HLS STREAM variable=stream_ddB depth=64
    #pragma HLS STREAM variable=stream_dC depth=64
    #pragma HLS STREAM variable=stream_yy0 depth=64

    DTYPE kernel_buffer[K];

#pragma HLS ARRAY_PARTITION variable=kernel_buffer type=complete dim=1

   load_kernel: for (int i = 0; i < K; i++) {
#pragma HLS PIPELINE II=1
        kernel_buffer[i] = kernel[i];
    }
    


    //intermediate register  
    DTYPE_VEC dd[VEC_N], dA[VEC_N], dB[VEC_N], dC[VEC_N], dX[VEC_N], dZ[VEC_N];
    DTYPE_VEC ddA[VEC_N], ddX[VEC_N], ddB[VEC_N], yy1[VEC_N];
    DTYPE_VEC yy0[VEC_N], Y[VEC_N];

     init_yy0: for(int j = 0; j < VEC_N; j++) {
        #pragma HLS PIPELINE II=1
        DTYPE_VEC zero_vec;
        for(int k = 0; k < VEC_FACTOR; k++) {
            #pragma HLS UNROLL
            zero_vec[k] = 0;
        }
        yy0[j] = zero_vec;
    }
    DTYPE_VEC delta_bias[VEC_FACTOR];
    EAU(delta, bias, delta_bias);
    softplus(delta_bias, dd);
    conv1d_vec(X, kernel_buffer, dX);
    silu(B, dB);
    silu(C, dC);
    silu(dX, ddX);
    silu(Z, dZ);
    EMU(dB, dd, ddB);
    EMU(A, dd, dA);
    EMU(ddX, D, yy1);
    exp1(dA, ddA);

    write_ddA: for(int j = 0; j < VEC_N; j++) {
        #pragma HLS PIPELINE II=1
        stream_ddA.write(ddA[j]);
    }
    
    write_ddX: for(int j = 0; j < VEC_N; j++) {
        #pragma HLS PIPELINE II=1
        stream_ddX.write(ddX[j]);
    }
    
    write_ddB: for(int j = 0; j < VEC_N; j++) {
        #pragma HLS PIPELINE II=1
        stream_ddB.write(ddB[j]);
    }
    
    write_dC: for(int j = 0; j < VEC_N; j++) {
        #pragma HLS PIPELINE II=1
        stream_dC.write(dC[j]);
    }
    
    write_yy0: for(int j = 0; j < VEC_N; j++) {
        #pragma HLS PIPELINE II=1
        stream_yy0.write(yy0[j]);
    }

//     update_H: for(int i=0; i<M; i++){
// #pragma HLS PIPELINE
// DTYPE_VEC hh0[VEC_N], hh1[VEC_N], dH[VEC_N];
//     	EMU(H0[i], ddA, &hh0[i]);
//     	EMU(ddX, ddB, &hh1[i]);
//     	EAU(&hh0[i], &hh1[i], H1[i]);
//     	EMU(dC, H1[i], &dH[i]);
//         accumulate(yy0, dH);
        
//     }
 update_H: for(int i = 0; i < M; i++) {
        #pragma HLS PIPELINE II=1
        
        for(int j = 0; j < VEC_N; j++) {
            #pragma HLS PIPELINE II=1
            
            DTYPE_VEC ddA_val = stream_ddA.read();
            DTYPE_VEC ddX_val = stream_ddX.read();
            DTYPE_VEC ddB_val = stream_ddB.read();
            DTYPE_VEC dC_val = stream_dC.read();
            DTYPE_VEC yy0_val = stream_yy0.read();

            DTYPE_VEC hh0, hh1, H1_val, dH_val;
            hh0 = H0[i][j] * ddA_val;
            hh1 = ddX_val * ddB_val;
            H1_val = hh0 + hh1;
            H1[i][j] = H1_val;
            
            dH_val = dC_val * H1_val;
            DTYPE_VEC new_yy0 = yy0_val + dH_val;
            
            stream_yy0.write(new_yy0);
        }
        
        if (i < M - 1) {
            refill_streams: for(int j = 0; j < VEC_N; j++) {
                #pragma HLS PIPELINE II=1
                stream_ddA.write(ddA[j]);
                stream_ddX.write(ddX[j]);
                stream_ddB.write(ddB[j]);
                stream_dC.write(dC[j]);
            }
        }
    }

read_final_yy0: for(int j = 0; j < VEC_N; j++) {
        #pragma HLS PIPELINE II=1
        yy0[j] = stream_yy0.read();
    }

    EAU(yy0, yy1, Y);
    EMU(Y, dZ, out);
}
