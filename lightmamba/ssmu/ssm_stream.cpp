#include "lightmamba.h"
#include <hls_stream.h>

static void load_vec(const DTYPE *in, hls::stream<DTYPE> &s, int n) {
#pragma HLS INLINE off
load_loop:
    for (int i=0;i<n;i++){
#pragma HLS PIPELINE II=1
        s.write(in[i]);
    }
}

static void store_vec(hls::stream<DTYPE> &s, DTYPE *out, int n) {
#pragma HLS INLINE off
store_loop:
    for (int i=0;i<n;i++){
#pragma HLS PIPELINE II=1
        out[i] = s.read();
    }
}

static void silu_stream(hls::stream<DTYPE> &in, hls::stream<DTYPE> &out, int n){
#pragma HLS INLINE off
silu_loop:
    for(int i=0;i<n;i++){
#pragma HLS PIPELINE II=1
        DTYPE x = in.read();
        DTYPE y = (DTYPE)(1.0 / (1.0 + hls::exp(-x)));
        out.write(x*y);
    }
}

static void exp_stream(hls::stream<DTYPE> &in, hls::stream<DTYPE> &out, int n){
#pragma HLS INLINE off
exp_loop:
    for(int i=0;i<n;i++){
#pragma HLS PIPELINE II=1
        out.write( (DTYPE)hls::exp(in.read()) );
    }
}

static void conv1d_stream(hls::stream<DTYPE> &x,
                          const DTYPE kernel[K],
                          hls::stream<DTYPE> &y, int n){
#pragma HLS INLINE off
    DTYPE win[K];
#pragma HLS ARRAY_PARTITION variable=win complete dim=1
init_win:
    for(int j=0;j<K;j++){win[j]=0;}
conv_loop:
    for(int i=0;i<n;i++){
#pragma HLS PIPELINE II=1
        // shift
        for(int j=K-1;j>0;j--){
#pragma HLS UNROLL
            win[j]=win[j-1];
        }
        win[0]=x.read();
        if(i>=K-1){
            DTYPE sum=0;
#pragma HLS BIND_OP op=add impl=dsp
acc_loop:
            for(int j=0;j<K;j++){
#pragma HLS UNROLL
                sum += kernel[j]*win[j];
            }
            y.write(sum);
        }
    }
}

static void emu_stream(hls::stream<DTYPE> &a,
                       hls::stream<DTYPE> &b,
                       hls::stream<DTYPE> &o,
                       int n){
#pragma HLS INLINE off
emu_loop:
    for(int i=0;i<n;i++){
#pragma HLS PIPELINE II=1
        o.write(a.read()*b.read());
    }
}

static void eau_stream(hls::stream<DTYPE> &a,
                       hls::stream<DTYPE> &b,
                       hls::stream<DTYPE> &o,
                       int n){
#pragma HLS INLINE off
eau_loop:
    for(int i=0;i<n;i++){
#pragma HLS PIPELINE II=1
        o.write(a.read()+b.read());
    }
}

static void stream_to_buf(hls::stream<DTYPE> &s, DTYPE buf[], int n){
#pragma HLS INLINE off
to_buf:
    for(int i=0;i<n;i++){
#pragma HLS PIPELINE II=1
        buf[i]=s.read();
    }
}

void SSMU(
    DTYPE kernel[K], DTYPE A[N], DTYPE B[N], DTYPE C[N], DTYPE D[N],
    DTYPE X[N], DTYPE Z[N],
    DTYPE H0[M][N], DTYPE H1[M][N],
    DTYPE delta[N], DTYPE bias[N], DTYPE out[N])
{
#pragma HLS INTERFACE m_axi port=X      offset=slave bundle=data_in
#pragma HLS INTERFACE m_axi port=out    offset=slave bundle=data_out
#pragma HLS INTERFACE m_axi port=H0     offset=slave bundle=data_H
#pragma HLS INTERFACE m_axi port=H1     offset=slave bundle=data_H
#pragma HLS INTERFACE m_axi port=A      offset=slave bundle=data_W
#pragma HLS INTERFACE m_axi port=B      offset=slave bundle=data_W
#pragma HLS INTERFACE m_axi port=C      offset=slave bundle=data_W
#pragma HLS INTERFACE m_axi port=D      offset=slave bundle=data_W
#pragma HLS INTERFACE m_axi port=Z      offset=slave bundle=data_W
#pragma HLS INTERFACE m_axi port=delta  offset=slave bundle=data_W
#pragma HLS INTERFACE m_axi port=bias   offset=slave bundle=data_W
#pragma HLS INTERFACE m_axi port=kernel offset=slave bundle=data_W
#pragma HLS INTERFACE s_axilite register port=return bundle=ctrl

#pragma HLS DATAFLOW

    hls::stream<DTYPE> sA, sB, sC, sD, sX, sZ, sDelta, sBias;
    hls::stream<DTYPE> sDeltaPlusBias, sDD, sDB, sDA, sDX_raw, sDX, sDZ, sYY1, sDDA;
#pragma HLS STREAM variable=sA  depth=64
#pragma HLS STREAM variable=sB  depth=64
#pragma HLS STREAM variable=sC  depth=64
#pragma HLS STREAM variable=sD  depth=64
#pragma HLS STREAM variable=sX  depth=64
#pragma HLS STREAM variable=sZ  depth=64
#pragma HLS STREAM variable=sDelta depth=64
#pragma HLS STREAM variable=sBias  depth=64
#pragma HLS STREAM variable=sDeltaPlusBias depth=64
#pragma HLS STREAM variable=sDD   depth=64
#pragma HLS STREAM variable=sDB   depth=64
#pragma HLS STREAM variable=sDA   depth=64
#pragma HLS STREAM variable=sDX_raw depth=64
#pragma HLS STREAM variable=sDX   depth=64
#pragma HLS STREAM variable=sDZ   depth=64
#pragma HLS STREAM variable=sYY1  depth=64
#pragma HLS STREAM variable=sDDA  depth=64


    load_vec(A, sA, N);
    load_vec(B, sB, N);
    load_vec(C, sC, N);
    load_vec(D, sD, N);
    load_vec(X, sX, N);
    load_vec(Z, sZ, N);
    load_vec(delta, sDelta, N);
    load_vec(bias,  sBias,  N);

    // delta + bias -> softplus -> dd
    eau_stream(sDelta, sBias, sDeltaPlusBias, N);
    // softplus
    {
        // softplus(x) = log(1+exp(x))
        hls::stream<DTYPE> sExp, sOnePlus;
#pragma HLS STREAM variable=sExp depth=64
#pragma HLS STREAM variable=sOnePlus depth=64

        exp_stream(sDeltaPlusBias, sExp, N);
        // 1 + exp(x)
        for(int i=0;i<N;i++){
#pragma HLS PIPELINE II=1
            DTYPE t = sExp.read();
            sOnePlus.write((DTYPE)(1) + t);
        }
        // log
        for(int i=0;i<N;i++){
#pragma HLS PIPELINE II=1
            sDD.write( (DTYPE)hls::log( sOnePlus.read() ) );
        }
    }

    // silu(B) ⊙ dd  => ddB
    {
        hls::stream<DTYPE> sB_silu;
#pragma HLS STREAM variable=sB_silu depth=64
        silu_stream(sB, sB_silu, N);
        emu_stream(sB_silu, sDD, sDB, N);
    }

    // silu(C) 备用（如需）
    {
        hls::stream<DTYPE> sC_silu;
#pragma HLS STREAM variable=sC_silu depth=32
        silu_stream(sC, sC_silu, N);
    }
    conv1d_stream(sX, kernel, sDX_raw, N);
align_dx:
    for(int i=0;i<N;i++){
#pragma HLS PIPELINE II=1
        if(i < K-1) sDX.write((DTYPE)0);
        else        sDX.write(sDX_raw.read());
    }

    silu_stream(sZ, sDZ, N);

    // A ⊙ dd -> exp -> ddA
    {
        hls::stream<DTYPE> sDA_tmp;
#pragma HLS STREAM variable=sDA_tmp depth=64
        emu_stream(sA, sDD /*注意：此处 sDD 已被上游消费完；若需复用，请在上面复制一份*/, sDA_tmp, N);
        exp_stream(sDA_tmp, sDDA, N);
    }

    // ddX = silu(conv1d(X))
    {
        hls::stream<DTYPE> sDX_silu;
#pragma HLS STREAM variable=sDX_silu depth=64
        silu_stream(sDX, sDX_silu, N);
        // yy1 = ddX ⊙ D
        emu_stream(sDX_silu, sD, sYY1, N);
    }

    static DTYPE ddA_buf[N], ddB_buf[N], ddX_buf[N], dZ_buf[N], yy1_buf[N];
#pragma HLS BIND_STORAGE variable=ddA_buf type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=ddB_buf type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=ddX_buf type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=dZ_buf  type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=yy1_buf type=ram_1p impl=bram

    stream_to_buf(sDDA, ddA_buf, N);
    stream_to_buf(sDB,  ddB_buf, N);
    stream_to_buf(sYY1, yy1_buf, N);
    stream_to_buf(sDZ,  dZ_buf,  N);

    static DTYPE yy0[N];
#pragma HLS BIND_STORAGE variable=yy0 type=ram_1p impl=bram
init_yy0:
    for(int j=0;j<N;j++){
#pragma HLS PIPELINE II=1
        yy0[j]=0;
    }

row_loop:
    for(int i=0;i<M;i++){
#pragma HLS PIPELINE II=1
    col_loop:
        for(int j=0;j<N;j++){
#pragma HLS UNROLL factor=4
            DTYPE h0 = H0[i][j];
            DTYPE h1 = h0 * ddA_buf[j] + ddB_buf[j] * ddX_buf[j] + 0;
            H1[i][j] = h1;
            yy0[j]   += h1;
        }
    }

    // Y = yy0 + yy1_buf
    static DTYPE Y[N];
#pragma HLS BIND_STORAGE variable=Y type=ram_1p impl=bram
sum_y:
    for(int j=0;j<N;j++){
#pragma HLS PIPELINE II=1
        Y[j] = yy0[j] + yy1_buf[j];
    }

    // out = Y ⊙ dZ_buf
final_mul:
    for(int j=0;j<N;j++){
#pragma HLS PIPELINE II=1
        out[j] = Y[j] * dZ_buf[j];
    }
}
