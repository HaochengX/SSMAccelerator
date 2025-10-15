#include "ssmu.h"
void projection(DTYPE_VEC in[VEC_D], DTYPE_VEC weight[N][VEC_D], DTYPE_VEC out[N]) {
    projection_loop: for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        DTYPE sum[VEC_FACTOR] = {0};
        dot_product: for(int j = 0; j < VEC_D; j++) {
            #pragma HLS UNROLL
            for(int k = 0; k < VEC_FACTOR; k++) {
                #pragma HLS UNROLL
                sum[k] += in[j][k] * weight[i][j][k];
            }
        }
        for(int k = 0; k < VEC_FACTOR; k++) {
            #pragma HLS UNROLL
            out[i][k] = sum[k];
        }
    }
}

void projection_delta(DTYPE_VEC in[VEC_D], DTYPE_VEC weight[VEC_D][VEC_D], DTYPE_VEC out[VEC_D]) {
    projection_delta_loop: for(int i = 0; i < VEC_D; i++) {
        #pragma HLS PIPELINE II=1
        DTYPE sum[VEC_FACTOR] = {0};
        
        dot_product_delta: for(int j = 0; j < VEC_D; j++) {
            #pragma HLS UNROLL
            for(int k = 0; k < VEC_FACTOR; k++) {
                #pragma HLS UNROLL
                sum[k] += in[j][k] * weight[i][j][k];
            }
        }
        for(int k = 0; k < VEC_FACTOR; k++) {
            #pragma HLS UNROLL
            out[i][k] = (DTYPE)(hls::log(1 + hls::exp(sum[k])));
        }
    }
}

//silu
void silu(DTYPE_VEC in[VEC_D], DTYPE_VEC out[VEC_D]) {
    silu_vec_loop: for(int j = 0; j < VEC_D; j++) {
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

//exp
void exp1(DTYPE_VEC in[VEC_D], DTYPE_VEC out[VEC_D]) {
    exp1_loop: for(int j = 0; j < VEC_D; j++) {
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
void conv1d_vec(DTYPE_VEC input_X[VEC_D], DTYPE kernel[K], DTYPE_VEC Y[VEC_D]) {
    #pragma HLS INLINE OFF
    
    static DTYPE shift_reg[K + VEC_FACTOR - 1];
    #pragma HLS ARRAY_PARTITION variable=shift_reg complete
    
    init_shift_reg: for(int i = 0; i < K + VEC_FACTOR - 1; i++) {
        #pragma HLS UNROLL
        shift_reg[i] = 0;
    }
    
    process_vectors: for(int i = 0; i < VEC_D; i++) {
        #pragma HLS PIPELINE II=1
        
        DTYPE_VEC input_vec = input_X[i];
        DTYPE_VEC output_vec;
        
        update_shift_reg: for(int j = K + VEC_FACTOR - 2; j >= VEC_FACTOR; j--) {
            #pragma HLS UNROLL
            shift_reg[j] = shift_reg[j - VEC_FACTOR];
        }
        
        load_data: for(int j = 0; j < VEC_FACTOR; j++) {
            #pragma HLS UNROLL
            shift_reg[j] = input_vec[j];
        }
        
        compute_output: for(int j = 0; j < VEC_FACTOR; j++) {
            #pragma HLS UNROLL
            
            DTYPE sum = 0;
            conv_calc: for(int k = 0; k < K; k++) {
                #pragma HLS UNROLL
                sum += kernel[k] * shift_reg[j + k];
            }
            output_vec[j] = sum;
        }
        
        Y[i] = output_vec;
    }
}

void EMU_2D(DTYPE_VEC A[N][VEC_D], DTYPE_VEC B[VEC_D], DTYPE_VEC out[N][VEC_D]) {
    EMU_2D_loop_i: for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        EMU_2D_loop_j: for(int j = 0; j < VEC_D; j++) {
            #pragma HLS UNROLL
            out[i][j] = A[i][j] * B[j];
        }
    }
}
//accumulator
// void accumulate(DTYPE_VEC accumulator[VEC_N], DTYPE_VEC addend[VEC_N]) {
//     accumulate_loop: for(int j = 0; j < VEC_N; j++) {
//         accumulator[j] = accumulator[j] + addend[j];
//     }
// }

void UpdateH_producer(
    DTYPE_VEC ddA[N][VEC_D], DTYPE_VEC dX[VEC_D], DTYPE_VEC dB[N][VEC_D],
     DTYPE_VEC H0[N][VEC_D],
    hls::stream<DTYPE_VEC> &stream_ddA,
    hls::stream<DTYPE_VEC> &stream_dX,
    hls::stream<DTYPE_VEC> &stream_dB,
    hls::stream<DTYPE_VEC> &stream_H0_in) {
    #pragma HLS DATAFLOW
    #pragma HLS INLINE OFF
    write_const_streams: for(int j = 0; j < VEC_D; j++) {
        #pragma HLS PIPELINE II=1
        stream_dX.write(dX[j]); 
    }
    write_H0_stream: for(int i = 0; i < N; i++) {
        for(int j = 0; j < VEC_D; j++) {
            #pragma HLS PIPELINE II=1
            stream_H0_in.write(H0[i][j]);
            stream_ddA.write(ddA[i][j]);
            stream_dB.write(dB[i][j]);
        }
    }
}

void UpdateH_consumer(
    hls::stream<DTYPE_VEC> &stream_ddA,
    hls::stream<DTYPE_VEC> &stream_X_ssm,
    hls::stream<DTYPE_VEC> &stream_dB,
    hls::stream<DTYPE_VEC> &stream_H0_in,
    DTYPE_VEC H1[N][VEC_D]) {
    #pragma HLS DATAFLOW
    #pragma HLS INLINE OFF
    DTYPE_VEC dX_val[VEC_D];
    read_ddX: for(int j = 0; j < VEC_D; j++) {
        #pragma HLS PIPELINE II=1
        dX_val[j] = stream_X_ssm.read();
    }


    process_timesteps: for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        
        DTYPE_VEC H0_current[VEC_D];
        read_H0_current: for(int j = 0; j < VEC_D; j++) {
            #pragma HLS PIPELINE II=1
            H0_current[j] = stream_H0_in.read();
        }
            
        process_elements: for(int j = 0; j < VEC_D; j++) {
            #pragma HLS PIPELINE II=1
            
            DTYPE_VEC ddA_val = stream_ddA.read();
            DTYPE_VEC dB_val = stream_dB.read();
            
            // H_out(Dim,N) = H_in(Dim,N) * ddA(Dim,N) + dB(Dim,N) * X_ssm(Dim)
            DTYPE_VEC hh0 = H0_current[j] * ddA_val;
            DTYPE_VEC hh1 = dB_val * dX_val[j];
            DTYPE_VEC H1_val = hh0 + hh1;
            H1[i][j] = H1_val;
        }
    }

}

//top function
void SSMU(
//DTYPE_BLOCK16 weight_pack_16[N], //16 * 8bit = 128bit
    DTYPE kernel[K],
    DTYPE_VEC A[N][VEC_D], DTYPE_VEC W_B[N][VEC_D], DTYPE_VEC W_C[N][VEC_D],
    DTYPE_VEC X[VEC_D],
    DTYPE_VEC H0[N][VEC_D], DTYPE_VEC H1[N][VEC_D],
    DTYPE_VEC W_delta[VEC_D][VEC_D],
    DTYPE_VEC out[VEC_D]){

#pragma HLS INTERFACE m_axi port=X offset=slave bundle=data_in depth=VEC_D
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=data_out depth=VEC_D
#pragma HLS INTERFACE m_axi port=H0 offset=slave bundle=data_H depth=N*VEC_D
#pragma HLS INTERFACE m_axi port=H1 offset=slave bundle=data_H depth=N*VEC_D
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=data_W depth=N*VEC_D
#pragma HLS INTERFACE m_axi port=W_B offset=slave bundle=data_W depth=N*VEC_D
#pragma HLS INTERFACE m_axi port=W_C offset=slave bundle=data_W depth=N*VEC_D
#pragma HLS INTERFACE m_axi port=W_delta offset=slave bundle=data_W depth=VEC_D*VEC_D
#pragma HLS INTERFACE m_axi port=kernel offset=slave bundle=data_W depth=K
#pragma HLS INTERFACE s_axilite register port=return bundle=ctrl

    hls::stream<DTYPE_VEC> stream_ddA;
    hls::stream<DTYPE_VEC> stream_dX; 
    hls::stream<DTYPE_VEC> stream_dB;
    hls::stream<DTYPE_VEC> stream_C;
    hls::stream<DTYPE_VEC> stream_H0_in;
    hls::stream<DTYPE_VEC> stream_H1_out;
    
    #pragma HLS STREAM variable=stream_ddA depth=32
    #pragma HLS STREAM variable=stream_dX depth=32
    #pragma HLS STREAM variable=stream_dB depth=32
    #pragma HLS STREAM variable=stream_C depth=32
    #pragma HLS STREAM variable=stream_H0_in depth=32
    #pragma HLS STREAM variable=stream_H1_out depth=32

    DTYPE kernel_buffer[K];

#pragma HLS ARRAY_PARTITION variable=kernel_buffer type=complete dim=1

   load_kernel: for (int i = 0; i < K; i++) {
#pragma HLS PIPELINE II=1
        kernel_buffer[i] = kernel[i];
    }
    


    //intermediate register  
    DTYPE_VEC delta[VEC_D], dA[N][VEC_D], B[N], C[N], dX[VEC_D];
    DTYPE_VEC ddA[N][VEC_D], X_ssm[VEC_D], dB[N][VEC_D], X_gate[VEC_D];


    
    //linear X
    conv1d_vec(X, kernel_buffer, dX);
    silu(dX, X_ssm);
    silu(X,X_gate);
    //input_dependent projection
    projection(X_ssm, W_B, B);
    projection(X_ssm, W_C, C);
    projection_delta(X_ssm, W_delta, delta);
    // Calculate dA(Dim,N) = A(Dim,N) * delta(Dim)
    EMU_2D(A, delta, dA);
    
    // Calculate ddA(Dim,N) = exp(dA)
    exp_2d_loop: for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        exp1(dA[i], ddA[i]);
    }
    
    // Calculate dB(Dim,N) = B(N) * delta(Dim)
    calc_dB_loop_i: for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        calc_dB_loop_j: for(int j = 0; j < VEC_D; j++) {
            #pragma HLS UNROLL
            dB[i][j] = B[i] * delta[j];
        }
    }

    
UpdateH_producer(ddA, X_ssm, dB, H0, 
                     stream_ddA, stream_dX, stream_dB, stream_H0_in);
    
UpdateH_consumer(stream_ddA, stream_dX, stream_dB, stream_H0_in,
                     H1);
    init_out: for(int j = 0; j < VEC_D; j++) {
        #pragma HLS PIPELINE II=1
        out[j] = X_gate[j];
    }
    
    // H1(Dim,N) * dC(N)
    compute_h1_dC: for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        accumulate_loop: for(int j = 0; j < VEC_D; j++) {
            #pragma HLS UNROLL
            DTYPE_VEC h1_C = H1[i][j] * C[i];
            out[j] = out[j] + h1_C;
        }
    }
}
