#ifdef RMSNORM_H
#define RMSNORM_H
#include "Mamba.h"
template<int DIM>
inline void rms_norm_gated_core(
    hls::stream<DTYPE_VEC>& y_stream,      
    hls::stream<DTYPE_VEC>& z_stream,      
    hls::stream<DTYPE_VEC>& output_stream, 
    const FDTYPE eps 
) {
    
    SEQUENCE_LOOP: for (int seq = 0; seq < LENGTH; seq++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=1024
        
        FDTYPE square_sum = FDTYPE(0);
        DTYPE_VEC gated_buffer[VEC_I];
        
        for (int i = 0; i < VEC_I; i++) {
            #pragma HLS PIPELINE II=1
            
            DTYPE_VEC y_vec = y_stream.read();
            DTYPE_VEC z_vec = z_stream.read();
            
            DTYPE_VEC gated_vec;
            
            for (int v = 0; v < VEC_FACTOR; v++) {
                #pragma HLS UNROLL
                int global_idx = i * VEC_FACTOR + v;
                
                if (global_idx < DIM) {
                    FDTYPE z_val = (FDTYPE)z_vec[v];
                    FDTYPE silu_z = (FDTYPE)silu_elem(z_vec[v]);  
                    
                    FDTYPE y_val = (FDTYPE)y_vec[v];
                    FDTYPE gated_val = y_val * silu_z;
                    
                    gated_vec[v] = (DTYPE)gated_val;
                    
                    square_sum += gated_val * gated_val;
                } else {
                    gated_vec[v] = DTYPE(0);
                }
            }
            
            gated_buffer[i] = gated_vec;  
        }
        
        FDTYPE mean = square_sum / DIM;
        FDTYPE rms = hls::sqrt(mean + eps);
        FDTYPE scale = FDTYPE(1) / rms;
        
        for (int i = 0; i < VEC_I; i++) {
            #pragma HLS PIPELINE II=1
            
            DTYPE_VEC gated_vec = gated_buffer[i];
            DTYPE_VEC norm_vec;
            
            for (int v = 0; v < VEC_FACTOR; v++) {
                #pragma HLS UNROLL
                int global_idx = i * VEC_FACTOR + v;
                
                if (global_idx < DIM) {
                    // 归一化：除以RMS
                    FDTYPE gated_val = (FDTYPE)gated_vec[v];
                    FDTYPE norm_val = gated_val * scale;
                    
                    norm_vec[v] = (DTYPE)norm_val;
                } else {
                    norm_vec[v] = DTYPE(0);
                }
            }
            
            output_stream.write(norm_vec);
        }
    }
}



#endif 