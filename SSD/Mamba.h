#ifndef Mamba_H
#define Mamba_H

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <hls_vector.h>

typedef ap_fixed<32, 10> DTYPE;
constexpr int BATCH=1; 
constexpr int LENGTH=64;
constexpr int DIM=2560;//model dimension
constexpr int EXPAND=2;
constexpr int I=DIM*EXPAND;//inner
constexpr int N=128;//state dimension
constexpr int K=4;//conv kernel size
constexpr int CHUNK=64; //chunk
constexpr int H=80;//NHEAD
constexpr int P=64;//HEADDIM, H*P=INNER
constexpr int NUM_CHUNKS = LENGTH / CHUNK;

constexpr int INPUT_LINEAR_SIZE= 2*I + H + 2*N;
constexpr int CONV_DIM = I + 2*N;

constexpr int VEC_FACTOR=16; //VEC can be 1,2,4,8,16
typedef hls::vector<DTYPE, VEC_FACTOR> DTYPE_VEC;

constexpr int VEC_D=DIM / VEC_FACTOR;
constexpr int VEC_I=I / VEC_FACTOR;
constexpr int VEC_INPUT_LINEAR=INPUT_LINEAR_SIZE / VEC_FACTOR;
constexpr int VEC_N=N / VEC_FACTOR;
constexpr int VEC_H=H / VEC_FACTOR;
constexpr int VEC_CONV_DIM=CONV_DIM/VEC_FACTOR;
constexpr int VEC_PER_CHUNK = CHUNK / VEC_FACTOR;
constexpr int VEC_PER_HEAD = P / VEC_FACTOR;
constexpr int VEC_PER_STATE = (P * N) / VEC_FACTOR;

typedef float FDTYPE;

static inline DTYPE silu_elem(DTYPE a)
{
    FDTYPE x = (FDTYPE)a;
    FDTYPE expv = hls::exp(-x);
    FDTYPE sig = (FDTYPE)1.0 / ((FDTYPE)1.0 + expv);
    FDTYPE res = x * sig;
    // cast back to DTYPE (beware saturation if DTYPE is narrow)
    return (DTYPE)res;
}

static inline DTYPE exp_elem(DTYPE a)
{
    FDTYPE x = (FDTYPE)a;
    FDTYPE y = hls::exp(x);
    return (DTYPE)y;
}

static inline DTYPE softplus_elem(DTYPE a)
{
    FDTYPE x = (FDTYPE)a;
    FDTYPE y = hls::log((FDTYPE)1.0 + hls::exp(x));
    return (DTYPE)y;
}
#endif