#ifndef LUT_OPTIMIZED_H
#define LUT_OPTIMIZED_H

#include "ssmu.h"
// EXP LUT
constexpr int EXP_LOG2DENOM = 3;
constexpr int EXP_LOG2DENOM_S = 10;
constexpr int EXP_ENTRIES = 4096;
constexpr int EXP_ENTRIES_S = 32;
constexpr int EXP_ALPHA = -20480;

// Softplus LUT
constexpr int SOFTplus_LOG2DENOM = 3;
constexpr int SOFTplus_LOG2DENOM_S = 10;
constexpr int SOFTplus_ENTRIES = 4096;
constexpr int SOFTplus_ENTRIES_S = 32;
constexpr int SOFTplus_ALPHA = -20480;

// SiLU LUT
constexpr int SILU_LOG2DENOM = 3;
constexpr int SILU_LOG2DENOM_S = 10;
constexpr int SILU_ENTRIES = 4096;
constexpr int SILU_ENTRIES_S = 32;
constexpr int SILU_ALPHA = -20480;

// EXP 
constexpr int exp_table[] = {
    #include "exp_table.txt"
};

constexpr int exp_table_s[] = {
    #include "exp_table_s.txt"
};

// Softplus 
constexpr int softplus_table[] = {
    #include "softplus_table.txt"
};

constexpr int softplus_table_s[] = {
    #include "softplus_table_s.txt"
};

// SiLU 
constexpr int silu_table[] = {
    #include "silu_table.txt"  
};

constexpr int silu_table_s[] = {
    #include "silu_table_s.txt"  
};


template<typename T>
T clamp(T value, T min_val, T max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}
constexpr float FIXED_POINT_SCALE = 1024.0f;


inline DTYPE lut_exp(DTYPE x) {
    #pragma HLS INLINE
    int x_fixed = (int)(x * FIXED_POINT_SCALE);
    
    int LUT_IDX = (x_fixed - EXP_ALPHA) >> EXP_LOG2DENOM;
    int LUT_S_IDX = (x_fixed - EXP_ALPHA) >> EXP_LOG2DENOM_S;
    
    LUT_IDX = clamp(LUT_IDX, 0, EXP_ENTRIES - 1);
    LUT_S_IDX = clamp(LUT_S_IDX, 0, EXP_ENTRIES_S - 1);
    
    auto exp_val = exp_table[LUT_IDX];
    auto exp_s = exp_table_s[LUT_S_IDX];
    
    return (DTYPE)((exp_val << exp_s) / FIXED_POINT_SCALE);
}

inline DTYPE lut_softplus(DTYPE x) {
    #pragma HLS INLINE
    int x_fixed = (int)(x * FIXED_POINT_SCALE);
    
    int LUT_IDX = (x_fixed - SOFTplus_ALPHA) >> SOFTplus_LOG2DENOM;
    int LUT_S_IDX = (x_fixed - SOFTplus_ALPHA) >> SOFTplus_LOG2DENOM_S;
    
    LUT_IDX = clamp(LUT_IDX, 0, SOFTplus_ENTRIES - 1);
    LUT_S_IDX = clamp(LUT_S_IDX, 0, SOFTplus_ENTRIES_S - 1);
    
    auto softplus_val = softplus_table[LUT_IDX];
    auto softplus_s = softplus_table_s[LUT_S_IDX];
    
    return (DTYPE)((softplus_val << softplus_s) / FIXED_POINT_SCALE);
}

inline DTYPE lut_silu(DTYPE x) {
    #pragma HLS INLINE
    int x_fixed = (int)(x * FIXED_POINT_SCALE);
    
    int LUT_IDX = (x_fixed - SILU_ALPHA) >> SILU_LOG2DENOM;
    int LUT_S_IDX = (x_fixed - SILU_ALPHA) >> SILU_LOG2DENOM_S;
    
    LUT_IDX = clamp(LUT_IDX, 0, SILU_ENTRIES - 1);
    LUT_S_IDX = clamp(LUT_S_IDX, 0, SILU_ENTRIES_S - 1);
    
    auto silu_val = silu_table[LUT_IDX];
    auto silu_s = silu_table_s[LUT_S_IDX];
    
    return (DTYPE)((silu_val << silu_s) / FIXED_POINT_SCALE);
}

inline DTYPE_VEC lut_exp_vec(DTYPE_VEC x_vec) {
    #pragma HLS INLINE
    DTYPE_VEC result;
    for(int k = 0; k < VEC_FACTOR; k++) {
        #pragma HLS UNROLL
        result[k] = lut_exp(x_vec[k]);
    }
    return result;
}

inline DTYPE_VEC lut_softplus_vec(DTYPE_VEC x_vec) {
    #pragma HLS INLINE
    DTYPE_VEC result;
    for(int k = 0; k < VEC_FACTOR; k++) {
        #pragma HLS UNROLL
        result[k] = lut_softplus(x_vec[k]);
    }
    return result;
}

inline DTYPE_VEC lut_silu_vec(DTYPE_VEC x_vec) {
    #pragma HLS INLINE
    DTYPE_VEC result;
    for(int k = 0; k < VEC_FACTOR; k++) {
        #pragma HLS UNROLL
        result[k] = lut_silu(x_vec[k]);
    }
    return result;
}

#endif