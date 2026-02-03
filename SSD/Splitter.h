// Splitter.h
#ifndef SPLITTER_H
#define SPLITTER_H

#include "Mamba.h"
template<int SUM, int X, int Y, int Z>
void splitter(
    hls::stream<DTYPE_VEC>& in_stream,
    hls::stream<DTYPE_VEC>& a_stream,
    hls::stream<DTYPE_VEC>& b_stream,
    hls::stream<DTYPE>& c_stream
) {
    
    constexpr int A_START = 0;
    constexpr int A_END = X;
    
    constexpr int B_START = A_END;
    constexpr int B_END = B_START + Y;
    
    constexpr int C_START = B_END;
    constexpr int C_END = C_START + Z;
    
    static int element_counter = 0;
    static DTYPE_VEC temp_vec;
    static int vec_offset = 0;
    
    PROCESS_LOOP: for (int vec_idx = 0; vec_idx < SUM/VEC_FACTOR; vec_idx++) {
        #pragma HLS PIPELINE II=1
        
        DTYPE_VEC input_vec = in_stream.read();
        
        PROCESS_VEC_ELEMENTS: for (int v = 0; v < VEC_FACTOR; v++) {
            #pragma HLS UNROLL
            
            int global_idx = vec_idx * VEC_FACTOR + v;
            
            if (global_idx >= SUM) {
                continue; 
            }
            
            DTYPE element = input_vec[v];
            if (global_idx < A_END) {
                temp_vec[vec_offset] = element;
                vec_offset++;
                if (vec_offset == VEC_FACTOR) {
                    a_stream.write(temp_vec);
                    vec_offset = 0;
                }
            } 
            else if (global_idx < B_END) {
                temp_vec[vec_offset] = element;
                vec_offset++;
                
                if (vec_offset == VEC_FACTOR) {
                    b_stream.write(temp_vec);
                    vec_offset = 0;
                }
            }
            else if (global_idx < C_END) {
                c_stream.write(element);
            }
        }
    }
    
    if (vec_offset > 0) {
        for (int v = vec_offset; v < VEC_FACTOR; v++) {
            temp_vec[v] = DTYPE(0);
        }
        
        int last_global_idx = VEC_INPUT_LINEAR * VEC_FACTOR - 1;
        
        if (last_global_idx < A_END) {
            a_stream.write(temp_vec);
        } 
        else if (last_global_idx < B_END) {
            b_stream.write(temp_vec);
        }
    }
}

#endif // SPLITTER_H