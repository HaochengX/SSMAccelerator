// Conv1D.h
#ifndef CONV1D_H
#define CONV1D_H

#include "Mamba.h"

inline void conv1d_sequence(
    hls::stream<DTYPE_VEC> &XBC_in,
    hls::stream<DTYPE> &kernel_in,
    hls::stream<DTYPE_VEC> &X_out,
    hls::stream<DTYPE_VEC> &B_out,
    hls::stream<DTYPE_VEC> &C_out
) {
    #pragma HLS INLINE off
    
    const int XBC_VEC_COUNT = 132;        
    const int X_VEC_COUNT = 128;          
    const int B_VEC_COUNT = 2;            
    const int C_VEC_COUNT = 2;            
    
    BATCH_LOOP: for (int l = 0; l < LENGTH; l++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=32
        
        DTYPE_VEC X_buffer[X_VEC_COUNT];
        DTYPE_VEC B_buffer[B_VEC_COUNT];
        DTYPE_VEC C_buffer[C_VEC_COUNT];
        
        for (int i = 0; i < XBC_VEC_COUNT; ++i) {
            #pragma HLS PIPELINE II = 1
            DTYPE_VEC data = XBC_in.read();
            
            if (i < X_VEC_COUNT) {
                X_buffer[i] = data;
            } else if (i < X_VEC_COUNT + B_VEC_COUNT) {
                B_buffer[i - X_VEC_COUNT] = data;
            } else {
                C_buffer[i - X_VEC_COUNT - B_VEC_COUNT] = data;
            }
        }
        
        static DTYPE kernel_buffer[K];
        if (l == 0) {
            for (int i = 0; i < K; ++i) {
                #pragma HLS PIPELINE
                kernel_buffer[i] = kernel_in.read();
            }
        }
        
        static DTYPE line_buffer[K - 1][VEC_FACTOR];
        #pragma HLS BIND_STORAGE variable = line_buffer type = ram_s2p impl = bram
        
        INIT_LINE_BUFFER: if (l == 0) {
            for (int i = 0; i < K - 1; ++i) {
                for (int k = 0; k < VEC_FACTOR; ++k) {
                    #pragma HLS PIPELINE II = 1
                    line_buffer[i][k] = 0;
                }
            }
        }
        
        CONV_PROC: for (int i = 0; i < X_VEC_COUNT; ++i) {
            DTYPE_VEC in_vec = X_buffer[i];
            DTYPE window[K][VEC_FACTOR];
            
            for (int j = 0; j < K - 1; ++j) {
                for (int k = 0; k < VEC_FACTOR; ++k) {
                    #pragma HLS PIPELINE II = 1
                    window[j][k] = line_buffer[j][k];
                }
            }
            
            for (int k = 0; k < VEC_FACTOR; ++k) {
                #pragma HLS PIPELINE II = 1
                window[K - 1][k] = in_vec[k];
            }
            
            for (int j = K - 2; j > 0; --j) {
                for (int k = 0; k < VEC_FACTOR; ++k) {
                    #pragma HLS PIPELINE II = 1
                    line_buffer[j][k] = line_buffer[j - 1][k];
                }
            }
            
            for (int k = 0; k < VEC_FACTOR; ++k) {
                #pragma HLS PIPELINE II = 1
                line_buffer[0][k] = in_vec[k];
            }
            
            DTYPE_VEC conv_out;
            for (int lane = 0; lane < VEC_FACTOR; ++lane) {
                FDTYPE sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    #pragma HLS PIPELINE II = 1
                    sum += (FDTYPE)kernel_buffer[k] * (FDTYPE)window[k][lane];
                }
                conv_out[lane] = (DTYPE)sum;
            }
            
            X_out.write(conv_out);
        }
        for (int i = 0; i < B_VEC_COUNT; ++i) {
            #pragma HLS PIPELINE II = 1
            B_out.write(B_buffer[i]);
        }
        
        for (int i = 0; i < C_VEC_COUNT; ++i) {
            #pragma HLS PIPELINE II = 1
            C_out.write(C_buffer[i]);
        }
    }
}

#endif // CONV1D_H