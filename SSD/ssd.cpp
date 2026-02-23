#include "Mamba.h"
#include "ssd_core.h"

void ssd_top(
    hls::stream<DTYPE_VEC>& X_stream,           
    hls::stream<DTYPE_VEC>& A_stream,           
    hls::stream<DTYPE_VEC>& B_stream,           
    hls::stream<DTYPE_VEC>& C_stream,           
    
    hls::stream<DTYPE_VEC>& initial_state_stream,  
    hls::stream<bool>& has_initial_state,          
    
    hls::stream<DTYPE_VEC>& Y_stream,         
    hls::stream<DTYPE_VEC>& final_state_stream 
) {
    #pragma HLS DATAFLOW
    
    hls::stream<DTYPE_VEC> A_cumsum_stream;
    hls::stream<DTYPE_VEC> L_stream;
    hls::stream<DTYPE_VEC> Y_diag_stream;
    hls::stream<DTYPE_VEC> states_stream;
    hls::stream<DTYPE_VEC> updated_states_stream;
    hls::stream<DTYPE_VEC> Y_off_stream;
    hls::stream<DTYPE_VEC> A_last_stream; 
    
    #pragma HLS STREAM variable=A_cumsum_stream depth=32
    #pragma HLS STREAM variable=L_stream depth=256
    #pragma HLS STREAM variable=Y_diag_stream depth=64
    #pragma HLS STREAM variable=states_stream depth=64
    #pragma HLS STREAM variable=updated_states_stream depth=64
    #pragma HLS STREAM variable=Y_off_stream depth=64
    #pragma HLS STREAM variable=A_last_stream depth=16
    
    hls::stream<DTYPE_VEC> A_rearranged_stream;
    #pragma HLS STREAM variable=A_rearranged_stream depth=32
    
    rearrange_A(A_stream, A_rearranged_stream);
    
    // 1. 
    cumsum(A_rearranged_stream, A_cumsum_stream);
    
    // 2. L（exp(segsum(A))）
    segsum_matrix(A_rearranged_stream, L_stream);
    
    // 3. Y_diag
    ssd_diag_block(X_stream, B_stream, C_stream, L_stream, Y_diag_stream);
    
    // 4. chunk state
    compute_state(A_cumsum_stream, B_stream, X_stream, states_stream);
    
    // 5. 
    extract_A_last(A_cumsum_stream, A_last_stream);
    inter_chunk_recurrence(A_last_stream, states_stream, initial_state_stream, 
                          has_initial_state, updated_states_stream, final_state_stream);
    
    // 6. Y_off
    state_to_output(C_stream, updated_states_stream, A_cumsum_stream, Y_off_stream);
    
    // 7. Y = Y_diag + Y_off
    combine_outputs(Y_diag_stream, Y_off_stream, Y_stream);
}