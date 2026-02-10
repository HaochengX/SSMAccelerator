// SSD_Core.h
#ifndef SSD_CORE_H
#define SSD_CORE_H

#include "Mamba.h"

inline void rearrange_for_ssd(
    hls::stream<DTYPE_VEC>& X_conv_stream,  
    hls::stream<DTYPE_VEC>& dA_stream,      
    hls::stream<DTYPE_VEC>& B_conv_stream, 
    hls::stream<DTYPE_VEC>& C_conv_stream,  
    
    hls::stream<DTYPE_VEC>& X_ssd_stream,  
    hls::stream<DTYPE_VEC>& dA_ssd_stream,  
    hls::stream<DTYPE_VEC>& B_ssd_stream,   
    hls::stream<DTYPE_VEC>& C_ssd_stream    
) {

        FDTYPE X_buffer[LENGTH][I];
        FDTYPE dA_buffer[LENGTH][H];
        FDTYPE B_buffer[LENGTH][N];
        FDTYPE C_buffer[LENGTH][N];
        
        
        READ_LOOP: for (int l = 0; l < LENGTH; l++) {
            for (int i_vec = 0; i_vec < VEC_I; i_vec++) {
                #pragma HLS PIPELINE II=1
                DTYPE_VEC x_vec = X_conv_stream.read();
                for (int v = 0; v < VEC_FACTOR; v++) {
                    #pragma HLS UNROLL
                    int i_idx = i_vec * VEC_FACTOR + v;
                    if (i_idx < I) {
                        X_buffer[l][i_idx] = (FDTYPE)x_vec[v];
                    }
                }
            }
            
            for (int h_vec = 0; h_vec < VEC_H; h_vec++) {
                #pragma HLS PIPELINE II=1
                DTYPE_VEC da_vec = dA_stream.read();
                for (int v = 0; v < VEC_FACTOR; v++) {
                    #pragma HLS UNROLL
                    int h_idx = h_vec * VEC_FACTOR + v;
                    if (h_idx < H) {
                        dA_buffer[l][h_idx] = (FDTYPE)da_vec[v];
                    }
                }
            }
            
            for (int n_vec = 0; n_vec < VEC_N; n_vec++) {
                #pragma HLS PIPELINE II=1
                DTYPE_VEC b_vec = B_conv_stream.read();
                for (int v = 0; v < VEC_FACTOR; v++) {
                    #pragma HLS UNROLL
                    int n_idx = n_vec * VEC_FACTOR + v;
                    if (n_idx < N) {
                        B_buffer[l][n_idx] = (FDTYPE)b_vec[v];
                    }
                }
            }
            
            for (int n_vec = 0; n_vec < VEC_N; n_vec++) {
                #pragma HLS PIPELINE II=1
                DTYPE_VEC c_vec = C_conv_stream.read();
                for (int v = 0; v < VEC_FACTOR; v++) {
                    #pragma HLS UNROLL
                    int n_idx = n_vec * VEC_FACTOR + v;
                    if (n_idx < N) {
                        C_buffer[l][n_idx] = (FDTYPE)c_vec[v];
                    }
                }
            }
        }
        
        for (int head = 0; head < H; head++) {
            for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
                for (int l_in_chunk = 0; l_in_chunk < VEC_PER_CHUNK; l_in_chunk++) {
                    #pragma HLS PIPELINE II=1
                    DTYPE_VEC da_vec;
                    int base_l = chunk * CHUNK + l_in_chunk * VEC_FACTOR;
                    
                    for (int v = 0; v < VEC_FACTOR; v++) {
                        #pragma HLS UNROLL
                        int l_idx = base_l + v;
                        if (l_idx < LENGTH) {
                            da_vec[v] = (DTYPE)dA_buffer[l_idx][head];
                        } else {
                            da_vec[v] = DTYPE(0);
                        }
                    }
                    
                    dA_ssd_stream.write(da_vec);
                }
            }
        }
        
        for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
            for (int l_in_chunk = 0; l_in_chunk < CHUNK; l_in_chunk++) {
                int l_idx = chunk * CHUNK + l_in_chunk;
                
                for (int head = 0; head < H; head++) {
                    #pragma HLS PIPELINE II=1
                    
                    DTYPE_VEC x_vec;
                    int base_p_idx = head * P;
                    
                    for (int p = 0; p < VEC_FACTOR; p++) {
                        #pragma HLS UNROLL
                        int p_idx = base_p_idx + p;
                        if (p_idx < I) {
                            x_vec[p] = (DTYPE)X_buffer[l_idx][p_idx];
                        } else {
                            x_vec[p] = DTYPE(0);
                        }
                    }
                    X_ssd_stream.write(x_vec);
                    
                    DTYPE_VEC b_vec;
                    for (int n = 0; n < VEC_FACTOR; n++) {
                        #pragma HLS UNROLL
                        if (n < N) {
                            b_vec[n] = (DTYPE)B_buffer[l_idx][n];
                        } else {
                            b_vec[n] = DTYPE(0);
                        }
                    }
                    B_ssd_stream.write(b_vec);
                    
                    DTYPE_VEC c_vec;
                    for (int n = 0; n < VEC_FACTOR; n++) {
                        #pragma HLS UNROLL
                        if (n < N) {
                            c_vec[n] = (DTYPE)C_buffer[l_idx][n];
                        } else {
                            c_vec[n] = DTYPE(0);
                        }
                    }
                    C_ssd_stream.write(c_vec);
                }
            }
        }
    }

inline void cumsum(
    hls::stream<DTYPE_VEC>& input_stream,
    hls::stream<DTYPE_VEC>& output_stream
) {
    
    SEQUENCE_LOOP: for (int seq_idx = 0; seq_idx < LENGTH; seq_idx++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=1024
        
        FDTYPE accumulator[VEC_FACTOR];
        
        for (int v = 0; v < VEC_FACTOR; v++) {
            #pragma HLS PIPELINE
            accumulator[v] = FDTYPE(0);
        }
        TIMESTEP_LOOP: for (int t = 0; t < VEC_PER_STATE; t++) {
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
    
    BATCH_LOOP: for (int batch = 0; batch < BATCH; batch++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=32
        
        SEQUENCE_LOOP: for (int seq_chunk = 0; seq_chunk < LENGTH / CHUNK; seq_chunk++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=16
            
            HEAD_LOOP: for (int head = 0; head < H; head++) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=8
                
                FDTYPE Bx_buffer[CHUNK][P];
                FDTYPE LBx_buffer[CHUNK][N];
                FDTYPE Y_buffer[CHUNK][P];
                
                
                COMPUTE_BX_LOOP: for (int i = 0; i < VEC_PER_CHUNK; i++) {
                    
                    DTYPE_VEC x_vec = x_stream.read();
                    DTYPE_VEC B_vec = B_stream.read();
                    
                    for (int v = 0; v < VEC_FACTOR; v++) {
                        #pragma HLS PIPELINE
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
                        FDTYPE sum = FDTYPE(0);
                        
                        for (int j = 0; j < CHUNK; j++) {
                            #pragma HLS PIPELINE
                            sum += L_row[j] * Bx_buffer[j][k];
                        }
                        
                        LBx_buffer[i][k] = sum;
                    }
                }
                
                COMPUTE_Y_DIAG_LOOP: for (int i = 0; i < VEC_PER_CHUNK; i++) {                    
                    DTYPE_VEC C_vec = C_stream.read();
                    DTYPE_VEC y_vec;
                    
                    for (int v = 0; v < VEC_FACTOR; v++) {
                        #pragma HLS PIPELINE
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

inline void compute_state(
    hls::stream<DTYPE_VEC>& A_cumsum_stream,  
    hls::stream<DTYPE_VEC>& B_stream,       
    hls::stream<DTYPE_VEC>& X_stream,       
    hls::stream<DTYPE_VEC>& state_stream   
) {
        CHUNK_LOOP: for (int chunk_idx = 0; chunk_idx < (LENGTH / CHUNK); chunk_idx++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=16
            
            HEAD_LOOP: for (int head = 0; head < H; head++) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=8
                
                FDTYPE A_cumsum_buffer[CHUNK];
                READ_A_CUMSUM_LOOP: for (int l = 0; l < VEC_PER_CHUNK; l++) {
                    #pragma HLS PIPELINE II=1
                    
                    DTYPE_VEC A_cumsum_vec = A_cumsum_stream.read();
                    
                    for (int v = 0; v < VEC_FACTOR; v++) {
                        #pragma HLS UNROLL
                        int element_idx = l * VEC_FACTOR + v;
                        
                        if (element_idx < CHUNK) {
                            A_cumsum_buffer[element_idx] = (FDTYPE)A_cumsum_vec[v];
                        }
                    }
                }
                
                FDTYPE A_cumsum_last = A_cumsum_buffer[CHUNK-1];
                
                FDTYPE decay_states_buffer[CHUNK];
                
                COMPUTE_DECAY_STATES_LOOP: for (int l = 0; l < CHUNK; l++) {
                    #pragma HLS PIPELINE II=1
                    FDTYPE diff = A_cumsum_last - A_cumsum_buffer[l];
                    decay_states_buffer[l] = hls::exp(diff);
                }
                
                FDTYPE state_buffer[P][N];
                
                INIT_STATE_LOOP: for (int p = 0; p < P; p++) {
                    for (int n = 0; n < N; n++) {
                        #pragma HLS PIPELINE
                        state_buffer[p][n] = FDTYPE(0);
                    }
                }
                
                COMPUTE_STATE_LOOP: for (int l = 0; l < VEC_PER_CHUNK; l++) {                    
                    DTYPE_VEC B_vec = B_stream.read();
                    DTYPE_VEC X_vec = X_stream.read();
                    
                    FDTYPE decay_val = decay_states_buffer[l];
                    
                    for (int v = 0; v < VEC_FACTOR; v++) {
                        int element_idx = l * VEC_FACTOR + v;
                        
                        if (element_idx < CHUNK) {
                            FDTYPE B_val = (FDTYPE)B_vec[v];
                            FDTYPE X_val = (FDTYPE)X_vec[v];
                            
                            for (int n = 0; n < N; n++) {
                                #pragma HLS PIPELINE
                                FDTYPE product = decay_val * B_val * X_val;
                                
                                for (int p = 0; p < P; p++) {
                                    #pragma HLS PIPELINE
                                    state_buffer[p][n] += product;
                                }
                            }
                        }
                    }
                }
                
                OUTPUT_STATE_LOOP: for (int n_block = 0; n_block < N / VEC_FACTOR; n_block++) {
                    #pragma HLS PIPELINE II=1
                    
                    for (int p = 0; p < P; p++) {
                        DTYPE_VEC state_vec;
                        
                        for (int v = 0; v < VEC_FACTOR; v++) {
                            #pragma HLS UNROLL
                            int n_idx = n_block * VEC_FACTOR + v;
                            
                            if (n_idx < N) {
                                state_vec[v] = (DTYPE)state_buffer[p][n_idx];
                            } else {
                                state_vec[v] = DTYPE(0);
                            }
                        }
                        
                        state_stream.write(state_vec);
                    }
                }
            }
        }
    }

inline void inter_chunk_recurrence(
    hls::stream<DTYPE_VEC>& A_cumsum_stream,     
    hls::stream<DTYPE_VEC>& states_stream,        
    hls::stream<DTYPE_VEC>& initial_state_stream, 
    hls::stream<bool>& has_initial_state,         
    hls::stream<DTYPE_VEC>& updated_states_stream, 
    hls::stream<DTYPE_VEC>& final_state_stream    
) {

    
    
    FDTYPE A_cumsum_last[NUM_CHUNKS];
    
    READ_A_CUMSUM_LAST_LOOP: for (int chunk_idx = 0; chunk_idx < NUM_CHUNKS; chunk_idx++) {
        #pragma HLS PIPELINE II=1
        
        DTYPE_VEC A_vec = A_cumsum_stream.read();
        
        A_cumsum_last[chunk_idx] = (FDTYPE)A_vec[0];
    }
    
    FDTYPE padded_A[NUM_CHUNKS + 1];
    
    padded_A[0] = FDTYPE(0);
    
    for (int i = 0; i < NUM_CHUNKS; i++) {
        #pragma HLS PIPELINE II=1
        padded_A[i + 1] = A_cumsum_last[i];
    }
    
    FDTYPE segsum_buffer[NUM_CHUNKS + 1];
    
    segsum_buffer[0] = padded_A[0];
    for (int i = 1; i <= NUM_CHUNKS; i++) {
        #pragma HLS PIPELINE II=1
        segsum_buffer[i] = segsum_buffer[i-1] + padded_A[i];
    }
    
    FDTYPE decay_chunk[NUM_CHUNKS + 1];
    
    for (int i = 0; i <= NUM_CHUNKS; i++) {
        #pragma HLS PIPELINE II=1
        decay_chunk[i] = hls::exp(segsum_buffer[i]);
    }
    
    FDTYPE current_state[P][N];
    
    bool has_init = has_initial_state.read();
    
    if (has_init) {
        READ_INITIAL_STATE_LOOP: for (int vec_idx = 0; vec_idx < VEC_PER_STATE; vec_idx++) {
            #pragma HLS PIPELINE II=1
            
            DTYPE_VEC init_vec = initial_state_stream.read();
            
            for (int v = 0; v < VEC_FACTOR; v++) {
                #pragma HLS UNROLL
                int flat_idx = vec_idx * VEC_FACTOR + v;
                
                if (flat_idx < P * N) {
                    int p_idx = flat_idx / N;
                    int n_idx = flat_idx % N;
                    current_state[p_idx][n_idx] = (FDTYPE)init_vec[v];
                }
            }
        }
    } else {
        for (int p = 0; p < P; p++) {
            for (int n = 0; n < N; n++) {
                #pragma HLS PIPELINE
                current_state[p][n] = FDTYPE(0);
            }
        }
    }
    
    FDTYPE states_buffer[NUM_CHUNKS][P][N];
    
    READ_ALL_STATES_LOOP: for (int chunk_idx = 0; chunk_idx < NUM_CHUNKS; chunk_idx++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=16
        
        for (int vec_idx = 0; vec_idx < VEC_PER_STATE; vec_idx++) {
            #pragma HLS PIPELINE II=1
            
            DTYPE_VEC state_vec = states_stream.read();
            
            for (int v = 0; v < VEC_FACTOR; v++) {
                #pragma HLS UNROLL
                int flat_idx = vec_idx * VEC_FACTOR + v;
                
                if (flat_idx < P * N) {
                    int p_idx = flat_idx / N;
                    int n_idx = flat_idx % N;
                    states_buffer[chunk_idx][p_idx][n_idx] = (FDTYPE)state_vec[v];
                }
            }
        }
    }
    for (int z = 0; z <= NUM_CHUNKS; z++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=17
        
        FDTYPE new_state[P][N];
        
        for (int p = 0; p < P; p++) {
            for (int n = 0; n < N; n++) {
                #pragma HLS PIPELINE II=1
                new_state[p][n] = FDTYPE(0);
            }
        }
        for (int c = 0; c < NUM_CHUNKS; c++) {
            
            FDTYPE decay_val = decay_chunk[z] * decay_chunk[c]; 
            
            for (int p = 0; p < P; p++) {
                for (int n = 0; n < N; n++) {
                    #pragma HLS PIPELINE II=1
                    new_state[p][n] += decay_val * states_buffer[c][p][n];
                }
            }
        }
        if (z > 0 && z <= NUM_CHUNKS) {
            OUTPUT_UPDATED_STATE_LOOP: for (int vec_idx = 0; vec_idx < VEC_PER_STATE; vec_idx++) {
                #pragma HLS PIPELINE II=1
                
                DTYPE_VEC out_vec;
                
                for (int v = 0; v < VEC_FACTOR; v++) {
                    #pragma HLS UNROLL
                    int flat_idx = vec_idx * VEC_FACTOR + v;
                    
                    if (flat_idx < P * N) {
                        int p_idx = flat_idx / N;
                        int n_idx = flat_idx % N;
                        out_vec[v] = (DTYPE)new_state[p_idx][n_idx];
                    } else {
                        out_vec[v] = DTYPE(0);
                    }
                }
                
                updated_states_stream.write(out_vec);
            }
        }
        
        if (z == NUM_CHUNKS) {
            OUTPUT_FINAL_STATE_LOOP: for (int vec_idx = 0; vec_idx < VEC_PER_STATE; vec_idx++) {
                #pragma HLS PIPELINE II=1
                
                DTYPE_VEC final_vec;
                
                for (int v = 0; v < VEC_FACTOR; v++) {
                    #pragma HLS UNROLL
                    int flat_idx = vec_idx * VEC_FACTOR + v;
                    
                    if (flat_idx < P * N) {
                        int p_idx = flat_idx / N;
                        int n_idx = flat_idx % N;
                        final_vec[v] = (DTYPE)new_state[p_idx][n_idx];
                    } else {
                        final_vec[v] = DTYPE(0);
                    }
                }
                
                final_state_stream.write(final_vec);
            }
        }
    }
}

inline void rearrange_A(
    hls::stream<DTYPE_VEC>& A_in_stream, 
    hls::stream<DTYPE_VEC>& A_out_stream 
) {
        FDTYPE A_buffer[H][LENGTH / CHUNK][CHUNK];
        
        for (int head = 0; head < H; head++) {
            for (int chunk = 0; chunk < (LENGTH / CHUNK); chunk++) {
                for (int l = 0; l < VEC_PER_CHUNK; l++) {
                    #pragma HLS PIPELINE II=1
                    
                    DTYPE_VEC A_vec = A_in_stream.read();
                    
                    for (int v = 0; v < VEC_FACTOR; v++) {
                        #pragma HLS UNROLL
                        int element_idx = l * VEC_FACTOR + v;
                        if (element_idx < CHUNK) {
                            A_buffer[head][chunk][element_idx] = (FDTYPE)A_vec[v];
                        }
                    }
                }
            }
        }
        
        for (int chunk = 0; chunk < (LENGTH / CHUNK); chunk++) {
            for (int l = 0; l < CHUNK; l++) {
                for (int head = 0; head < H; head++) {
                    #pragma HLS PIPELINE II=1
                    
                    DTYPE_VEC A_out_vec;
                    
                    for (int v = 0; v < VEC_FACTOR; v++) {
                        #pragma HLS UNROLL
                        int head_idx = head * VEC_FACTOR + v;
                        if (head_idx < H) {
                            A_out_vec[v] = (DTYPE)A_buffer[head_idx][chunk][l];
                        } else {
                            A_out_vec[v] = DTYPE(0);
                        }
                    }
                    
                    A_out_stream.write(A_out_vec);
                }
            }
        }
    }
    
inline void extract_A_last(
    hls::stream<DTYPE_VEC>& A_cumsum_stream, 
    hls::stream<DTYPE_VEC>& A_last_stream    
) {


        
        for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=16
            
            for (int l = 0; l < VEC_PER_CHUNK; l++) {
                #pragma HLS PIPELINE II=1
                
                DTYPE_VEC A_vec = A_cumsum_stream.read();
                
                if (l == VEC_PER_CHUNK - 1) {
                    DTYPE_VEC A_last_vec;
                    for (int v = 0; v < VEC_FACTOR; v++) {
                        #pragma HLS UNROLL
                        if (v == VEC_FACTOR - 1) {
                            A_last_vec[0] = A_vec[v]; 
                        }
                    }
                    
                    for (int head = 0; head < H; head++) {
                        #pragma HLS PIPELINE II=1
                        A_last_stream.write(A_last_vec);
                    }
                }
            }
        }
    }

inline void combine_outputs(
    hls::stream<DTYPE_VEC>& Y_diag_stream, 
    hls::stream<DTYPE_VEC>& Y_off_stream,
    hls::stream<DTYPE_VEC>& Y_stream 
) {

        for (int chunk = 0; chunk < (LENGTH / CHUNK); chunk++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=16
            
            for (int l = 0; l < VEC_PER_CHUNK; l++) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=64
                
                for (int head = 0; head < H; head++) {
                    for (int p_block = 0; p_block < VEC_PER_HEAD; p_block++) {
                        #pragma HLS PIPELINE II=1
                        
                        DTYPE_VEC Y_diag_vec = Y_diag_stream.read();
                        DTYPE_VEC Y_off_vec = Y_off_stream.read();
                        DTYPE_VEC Y_vec;
                        
                        // Y = Y_diag + Y_off
                        for (int v = 0; v < VEC_FACTOR; v++) {
                            #pragma HLS UNROLL
                            Y_vec[v] = Y_diag_vec[v] + Y_off_vec[v];
                        }
                        
                        Y_stream.write(Y_vec);
                    }
                }
            }
        }
    }

inline void state_to_output(
    hls::stream<DTYPE_VEC>& C_stream,          
    hls::stream<DTYPE_VEC>& states_stream,       
    hls::stream<DTYPE_VEC>& A_cumsum_stream,    
    hls::stream<DTYPE_VEC>& Y_off_stream       
) {
        
        CHUNK_LOOP: for (int chunk_idx = 0; chunk_idx < (LENGTH / CHUNK); chunk_idx++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=16
            
            FDTYPE states_buffer[H][P][N];
            #pragma HLS ARRAY_PARTITION variable=states_buffer cyclic factor=2 dim=3
            #pragma HLS ARRAY_PARTITION variable=states_buffer cyclic factor=2 dim=2
            #pragma HLS ARRAY_PARTITION variable=states_buffer cyclic factor=2 dim=1
            
            for (int head = 0; head < H; head++) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=8
                
                for (int vec_idx = 0; vec_idx < VEC_PER_STATE; vec_idx++) {
                    
                    DTYPE_VEC state_vec = states_stream.read();
                    
                    for (int v = 0; v < VEC_FACTOR; v++) {
                        #pragma HLS PIPELINE II=1
                        int flat_idx = vec_idx * VEC_FACTOR + v;
                        
                        if (flat_idx < P * N) {
                            int p_idx = flat_idx / N;
                            int n_idx = flat_idx % N;
                            states_buffer[head][p_idx][n_idx] = (FDTYPE)state_vec[v];
                        }
                    }
                }
            }
            
            POSITION_LOOP: for (int l = 0; l < VEC_PER_CHUNK; l++) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=64
                
                HEAD_LOOP2: for (int head = 0; head < H; head++) {
                    
                    DTYPE_VEC C_vec = C_stream.read();
                    
                    DTYPE_VEC A_vec = A_cumsum_stream.read();
                    FDTYPE state_decay_out = hls::exp((FDTYPE)A_vec[0]);
                    
                    DTYPE_VEC Y_off_vec;
                    
                    for (int p_v = 0; p_v < VEC_FACTOR; p_v++) {
                        #pragma HLS PIPELINE II=1
                        int p_idx = p_v; 
                        
                        if (p_idx < P) {
                            FDTYPE sum = FDTYPE(0);
                            
                            for (int n = 0; n < N; n++) {
                                #pragma HLS UNROLL factor=2
                                
                                FDTYPE C_val;
                                if (n < VEC_FACTOR) {
                                    C_val = (FDTYPE)C_vec[n];
                                } else {
                                    C_val = FDTYPE(0);
                                }
                                
                                FDTYPE state_val = states_buffer[head][p_idx][n];
                                sum += C_val * state_val;
                            }
                            
                            sum *= state_decay_out;
                            Y_off_vec[p_v] = (DTYPE)sum;
                        } else {
                            Y_off_vec[p_v] = DTYPE(0);
                        }
                    }
                    
                    Y_off_stream.write(Y_off_vec);
                }
            }
        }
    }

#endif // SSD_CORE_H