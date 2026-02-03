// Conv1D.h
#ifndef CONV1D_H
#define CONV1D_H

#include "Mamba.h"

inline void conv1d(
    hls::stream<DTYPE_VEC>& input_stream,
    
    const FDTYPE conv_weights[VEC_CONV_DIM][VEC_FACTOR][K],
    const FDTYPE conv_bias[VEC_CONV_DIM][VEC_FACTOR],
    
    hls::stream<DTYPE_VEC>& output_stream,
    
    int seq_len,
    bool
 use_silu
) {
        constexpr int NUM_GROUPS = VEC_CONV_DIM;
    
    // 使用循环缓冲区存储状态
    static FDTYPE group_states[NUM_GROUPS][VEC_FACTOR][K];
    static int head_pointers[NUM_GROUPS][VEC_FACTOR];
    
    #pragma HLS ARRAY_PARTITION variable=group_states complete dim=1
    #pragma HLS ARRAY_PARTITION variable=group_states complete dim=2
    #pragma HLS ARRAY_PARTITION variable=head_pointers complete dim=1
    #pragma HLS ARRAY_PARTITION variable=head_pointers complete dim=2
    
    bool initialized = false;
    
    TIMESTEP_LOOP: for (int t = 0; t < seq_len; t++) {
        #pragma HLS LOOP_TRIPCOUNT min=64 max=1024
        
        if (!initialized && t == 0) {
            INIT_HEAD: for (int g = 0; g < NUM_GROUPS; g++) {
                for (int ch = 0; ch < VEC_FACTOR; ch++) {
                    #pragma HLS UNROLL
                    head_pointers[g][ch] = 0;

                    for (int k = 0; k < K; k++) {
                        #pragma HLS UNROLL
                        group_states[g][ch][k] = 0;
                    }
                }
            }
            initialized = true;
        }
        
        GROUP_LOOP: for (int group = 0; group < NUM_GROUPS; group++) {
            #pragma HLS PIPELINE II=1
            
            DTYPE_VEC input_vec = input_stream.read();
            
            DTYPE_VEC output_vec;
            
            CHANNEL_LOOP: for (int ch = 0; ch < VEC_FACTOR; ch++) {
                #pragma HLS UNROLL
                
                FDTYPE input_val = (FDTYPE)input_vec[ch];
                
                int head = head_pointers[group][ch];
                
                group_states[group][ch][head] = input_val;
                
                FDTYPE conv_acc = conv_bias[group][ch];
                
                for (int k = 0; k < K; k++) {
                    #pragma HLS UNROLL
                    int state_idx = (head + k) % K;
                    conv_acc += group_states[group][ch][state_idx] * 
                               conv_weights[group][ch][k];
                }
                
                head_pointers[group][ch] = (head + 1) % K;
                
                if (use_silu) {
                    output_vec[ch] = (DTYPE)silu_elem(conv_acc);
                } else {
                    output_vec[ch] = (DTYPE)conv_acc;
                }
            }
            
            output_stream.write(output_vec);
        }
    }
}

#endif // MAMBA_CONV1D_OPTIMIZED_H