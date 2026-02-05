// SSD_Core.h
#ifndef SSD_CORE_H
#define SSD_CORE_H

#include "Mamba.h"

inline void cumsum(
    hls::stream<DTYPE_VEC>& input_stream,
    hls::stream<DTYPE_VEC>& output_stream
) {
    constexpr int VEC_PER_TIMESTEP = CHUNK / VEC_FACTOR;
    
    SEQUENCE_LOOP: for (int seq_idx = 0; seq_idx < LENGTH; seq_idx++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=1024
        
        FDTYPE accumulator[VEC_FACTOR];
        #pragma HLS ARRAY_PARTITION variable=accumulator complete dim=1
        
        for (int v = 0; v < VEC_FACTOR; v++) {
            #pragma HLS UNROLL
            accumulator[v] = FDTYPE(0);
        }
        TIMESTEP_LOOP: for (int t = 0; t < VEC_PER_TIMESTEP; t++) {
            #pragma HLS PIPELINE II=1
            
            DTYPE_VEC input_vec = input_stream.read();
            DTYPE_VEC output_vec;
            
            for (int v = 0; v < VEC_FACTOR; v++) {
                #pragma HLS UNROLL
                int element_idx = t * VEC_FACTOR + v;
                
                if (element_idx < CHUNK) {
                    FDTYPE input_val = (FDTYPE)input_vec[v];
                    accumulator[v] += input_val;
                    output_vec[v] = (DTYPE)accumulator[v];
                } else {
                    output_vec[v] = DTYPE(0);
                }
            }
            
            output_stream.write(output_vec);
        }
    }
}

inline void segsum_matrix(
    hls::stream<DTYPE_VEC>& A_stream,
    hls::stream<DTYPE_VEC>& L_stream
) {
    constexpr int VEC_PER_TIMESTEP = CHUNK / VEC_FACTOR;
    
    FDTYPE A_buffer[CHUNK];
    #pragma HLS ARRAY_PARTITION variable=A_buffer cyclic factor=VEC_FACTOR dim=1
    
    SEQUENCE_LOOP: for (int seq_idx = 0; seq_idx < LENGTH; seq_idx++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=1024
        
        READ_A_LOOP: for (int t = 0; t < VEC_PER_TIMESTEP; t++) {
            #pragma HLS PIPELINE II=1
            
            DTYPE_VEC A_vec = A_stream.read();
            
            for (int v = 0; v < VEC_FACTOR; v++) {
                #pragma HLS UNROLL
                int element_idx = t * VEC_FACTOR + v;
                
                if (element_idx < CHUNK) {
                    A_buffer[element_idx] = (FDTYPE)A_vec[v];
                }
            }
        }
        
        FDTYPE prefix_sum[CHUNK + 1];
        #pragma HLS ARRAY_PARTITION variable=prefix_sum block factor=16 dim=1
        
        prefix_sum[0] = FDTYPE(0);
        COMPUTE_PREFIX_LOOP: for (int i = 1; i <= CHUNK; i++) {
            #pragma HLS PIPELINE II=1
            prefix_sum[i] = prefix_sum[i-1] + A_buffer[i-1];
        }
        
        COMPUTE_L_MATRIX_ROW_LOOP: for (int i = 0; i < CHUNK; i++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=64
            
            for (int t = 0; t < VEC_PER_TIMESTEP; t++) {
                #pragma HLS PIPELINE II=1
                
                DTYPE_VEC L_row_vec;
                
                for (int v = 0; v < VEC_FACTOR; v++) {
                    #pragma HLS UNROLL
                    int j = t * VEC_FACTOR + v;
                    
                    if (j < CHUNK) {
                        if (j <= i) {
                            FDTYPE seg_sum = prefix_sum[i] - prefix_sum[j];
                            
                            L_row_vec[v] = (DTYPE)hls::exp(seg_sum);
                        } else {
                            L_row_vec[v] = DTYPE(0);
                        }
                    } else {
                        L_row_vec[v] = DTYPE(0);
                    }
                }
                L_stream.write(L_row_vec);
            }
        }
    }
}

inline void ssd_diag_block(
    hls::stream<DTYPE_VEC>& x_stream,    
    hls::stream<DTYPE_VEC>& B_stream,   
    hls::stream<DTYPE_VEC>& C_stream,    
    hls::stream<DTYPE_VEC>& L_stream,    
    hls::stream<DTYPE_VEC>& y_diag_stream
) {
    constexpr int VEC_PER_CHUNK = CHUNK / VEC_FACTOR;
    
    BATCH_LOOP: for (int batch = 0; batch < BATCH; batch++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=32
        
        SEQUENCE_LOOP: for (int seq_chunk = 0; seq_chunk < LENGTH / CHUNK; seq_chunk++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=16
            
            HEAD_LOOP: for (int head = 0; head < H; head++) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=8
                
                FDTYPE Bx_buffer[CHUNK][P];
                FDTYPE LBx_buffer[CHUNK][N];
                FDTYPE Y_buffer[CHUNK][P];
                
                #pragma HLS ARRAY_PARTITION variable=Bx_buffer cyclic factor=2 dim=1
                #pragma HLS ARRAY_PARTITION variable=LBx_buffer cyclic factor=2 dim=1
                #pragma HLS ARRAY_PARTITION variable=Y_buffer cyclic factor=2 dim=1
                
                COMPUTE_BX_LOOP: for (int i = 0; i < VEC_PER_CHUNK; i++) {
                    #pragma HLS PIPELINE II=1
                    
                    DTYPE_VEC x_vec = x_stream.read();
                    DTYPE_VEC B_vec = B_stream.read();
                    
                    for (int v = 0; v < VEC_FACTOR; v++) {
                        #pragma HLS UNROLL
                        int idx = i * VEC_FACTOR + v;
                        
                        if (idx < CHUNK) {
                            FDTYPE x_val = (FDTYPE)x_vec[v];
                            FDTYPE B_val = (FDTYPE)B_vec[v];
                            
                            for (int d = 0; d < P; d++) {
                                #pragma HLS UNROLL factor=2
                                Bx_buffer[idx][d] = B_val * x_val;
                            }
                        }
                    }
                }
                
                COMPUTE_LBX_LOOP: for (int i = 0; i < CHUNK; i++) {
                    #pragma HLS PIPELINE II=1
                    
                    FDTYPE L_row[CHUNK];
                    
                    for (int j = 0; j < VEC_PER_CHUNK; j++) {
                        #pragma HLS PIPELINE II=1
                        DTYPE_VEC L_vec = L_stream.read();
                        
                        for (int v = 0; v < VEC_FACTOR; v++) {
                            #pragma HLS UNROLL
                            int col_idx = j * VEC_FACTOR + v;
                            if (col_idx < CHUNK) {
                                L_row[col_idx] = (FDTYPE)L_vec[v];
                            }
                        }
                    }
                    
                    for (int k = 0; k < N; k++) {
                        #pragma HLS UNROLL factor=2
                        FDTYPE sum = FDTYPE(0);
                        
                        for (int j = 0; j < CHUNK; j++) {
                            #pragma HLS UNROLL factor=2
                            sum += L_row[j] * Bx_buffer[j][k];
                        }
                        
                        LBx_buffer[i][k] = sum;
                    }
                }
                
                COMPUTE_Y_DIAG_LOOP: for (int i = 0; i < VEC_PER_CHUNK; i++) {
                    #pragma HLS PIPELINE II=1
                    
                    DTYPE_VEC C_vec = C_stream.read();
                    DTYPE_VEC y_vec;
                    
                    for (int v = 0; v < VEC_FACTOR; v++) {
                        #pragma HLS UNROLL
                        int idx = i * VEC_FACTOR + v;
                        
                        if (idx < CHUNK) {
                            FDTYPE C_val = (FDTYPE)C_vec[v];
                            FDTYPE sum = FDTYPE(0);
                            
                            for (int k = 0; k < N; k++) {
                                #pragma HLS UNROLL factor=2
                                sum += C_val * LBx_buffer[idx][k];
                            }
                            
                            y_vec[v] = (DTYPE)sum;
                        } else {
                            y_vec[v] = DTYPE(0);
                        }
                    }
                    
                    y_diag_stream.write(y_vec);
                }
            }
        }
    }
}

#endif // SSD_CORE_H