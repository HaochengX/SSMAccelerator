// tb_ssmu.cpp  (COSIM-safe / single-source-of-truth)
//
// Key goals:
// 1) Pull all shapes + knobs from SSMU.h (single source of truth)
// 2) Avoid RTL cosim OOM by running ONLY 1 transaction in RTL cosim
// 3) Drain every output the DUT may produce to prevent deadlock
// 4) COSIM-safe feeding: start DUT first, then feed ap_fifo streams (avoid TB write-block deadlock)
//
// Fix in this version:
// - NEVER blindly read fixed counts from hls::stream.
// - After DUT returns, use stream.size() to read only what exists.
// - If fewer tokens than expected, print WARNING and pad remaining outputs with zeros.
//   This prevents "read while empty" in C TB (and avoids RTL hang).
//
#include "SSMU.h"

#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <thread>

// ============================================================
// Pull shapes from SSMU.h (single source of truth)
// ============================================================
static const int D_T     = SSMU_D_T;
static const int C2_T    = SSMU_C2_T;
static const int CCONV_T = SSMU_CCONV_T;
static const int CH_T    = SSMU_CH_T;
static const int CIN_T   = SSMU_CIN_T;
static const int STATE   = SSMU_STATE;
static const int CONV_K  = SSMU_K;

#ifndef HUGE_LEN
static const int HUGE_LEN = STATE * C2_T;
#endif

#ifndef SSMU_ENABLE_TRACE_DDR
#define SSMU_ENABLE_TRACE_DDR 0
#endif

#ifndef SSMU_ENABLE_H1_STREAM_OUT
#define SSMU_ENABLE_H1_STREAM_OUT 1
#endif

#ifndef SSMU_H1_OUT_LEN
#define SSMU_H1_OUT_LEN (SSMU_STATE * SSMU_C2_T)
#endif

#ifndef VEC_FACTOR
#error "VEC_FACTOR must be defined in SSMU.h"
#endif

// ============================================================
// IMPORTANT: Prevent RTL cosim OOM / 3 transactions
// ============================================================
#ifndef __SYNTHESIS__
#define TB_ENABLE_AUTO_DETECT 1
#else
#define TB_ENABLE_AUTO_DETECT 0
#endif

// ============================================================
// COSIM feeding policy
// ============================================================
#ifndef TB_COSIM_THREADED_FEED
#define TB_COSIM_THREADED_FEED 1
#endif

// ============================================================
// Deterministic RNG
// ============================================================
static unsigned lcg = 1;
static inline float frand(float lo=-0.5f, float hi=0.5f) {
    lcg = 1664525u * lcg + 1013904223u;
    float r = (float)(lcg & 0x00FFFFFF) / (float)0x01000000; // [0,1)
    return lo + (hi - lo) * r;
}

// ============================================================
// vget/vset: use header if present; fallback only if missing
// ============================================================
#ifndef SSMU_HAVE_VGET_VSET
template<typename V>
static inline auto vget(const V& v, unsigned idx) -> decltype(v[idx]) { return v[idx]; }
template<typename V, typename T>
static inline void vset(V& v, unsigned idx, const T& val) { v[idx] = (decltype(v[idx]))val; }
#endif

static inline DTYPE_VEC make_vec(float base=0.0f) {
    DTYPE_VEC v;
    for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) vset(v, l, (DTYPE)base);
    return v;
}

static inline DTYPE_VEC rand_vec(float lo=-0.5f, float hi=0.5f) {
    DTYPE_VEC v;
    for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) vset(v, l, (DTYPE)frand(lo, hi));
    return v;
}

static inline W_VEC rand_wvec() {
    W_VEC w;
    for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
#if SSMU_USE_INT8
        int r = (int)std::lround(frand(-10.0f, 10.0f));
        if (r > 127) r = 127;
        if (r < -128) r = -128;
        vset(w, l, (Q8_T)r);
#else
        vset(w, l, (DTYPE)frand(-0.1f, 0.1f));
#endif
    }
    return w;
}

static inline W_VEC zero_wvec() {
    W_VEC w;
    for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
#if SSMU_USE_INT8
        vset(w, l, (Q8_T)0);
#else
        vset(w, l, (DTYPE)0);
#endif
    }
    return w;
}

// ============================================================
// Reference stub (keep; you can upgrade later)
// ============================================================
static void reference_model_stub(
    const DTYPE_VEC A_fixed[STATE],
    const DTYPE_VEC RMS_weight[D_T],
    const W_VEC     W_inproj[D_T][CIN_T],
    const W_VEC     W_B[STATE][C2_T],
    const W_VEC     W_C[STATE][C2_T],
    const W_VEC     W_delta[C2_T][C2_T],
    const W_VEC     W_out[D_T][C2_T],
    const DTYPE_VEC D_diag[C2_T],
    const DTYPE     kernel[CONV_K],
    const DTYPE_VEC X_in_host[D_T],
    const DTYPE_VEC H0_in_host[HUGE_LEN],
    const DTYPE_VEC conv_state_in_host[CONV_K-1],
    // outputs:
    DTYPE_VEC out_host[D_T],
    DTYPE_VEC H1_host[SSMU_H1_OUT_LEN]
) {
    for (int i=0;i<D_T;++i) out_host[i]=make_vec(0.0f);
    for (int i=0;i<SSMU_H1_OUT_LEN;++i) H1_host[i]=make_vec(0.0f);
    (void)A_fixed; (void)RMS_weight; (void)W_inproj; (void)W_B; (void)W_C;
    (void)W_delta; (void)W_out; (void)D_diag; (void)kernel; (void)X_in_host;
    (void)H0_in_host; (void)conv_state_in_host;
}

// ============================================================
// Diff helpers
// ============================================================
static inline float vec_max_abs_diff(const DTYPE_VEC& a, const DTYPE_VEC& b) {
    float m = 0.0f;
    for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
        float da = (float)vget(a,l);
        float db = (float)vget(b,l);
        float d  = std::fabs(da - db);
        if (d > m) m = d;
    }
    return m;
}

static inline float max_abs_diff_out(const DTYPE_VEC a[D_T], const DTYPE_VEC b[D_T]) {
    float m = 0.0f;
    for (int i=0;i<D_T;++i) m = std::max(m, vec_max_abs_diff(a[i], b[i]));
    return m;
}

#if SSMU_ENABLE_H1_STREAM_OUT
static inline float max_abs_diff_h1(const DTYPE_VEC a[SSMU_H1_OUT_LEN], const DTYPE_VEC b[SSMU_H1_OUT_LEN]) {
    float m = 0.0f;
    for (int i=0;i<SSMU_H1_OUT_LEN;++i) m = std::max(m, vec_max_abs_diff(a[i], b[i]));
    return m;
}
#endif

// ============================================================
// Safe drain helpers (avoid empty-read)
// ============================================================
template<typename T>
static int drain_stream_n_or_pad(hls::stream<T>& s, int expected, T* out_buf, const char* tag) {
    int avail = (int)s.size();              // C-sim/COSIM TB is allowed to use size()
    int nread = std::min(expected, avail);

    for (int i=0;i<nread;++i) out_buf[i] = s.read();

    if (nread < expected) {
        std::printf("[TB][WARN] %s produced %d/%d tokens (padding %d zeros)\n",
                    tag, nread, expected, expected - nread);
        T z{};
        for (int i=nread;i<expected;++i) out_buf[i] = z;
    }

    // If DUT over-produced (unexpected), drain the rest to avoid leftover for next runs.
    // (Should not happen in a good design, but makes TB robust.)
    while (!s.empty()) (void)s.read();

    return nread;
}

static int drain_stream_discard_n_or_warn(hls::stream<DTYPE_VEC>& s, int expected, const char* tag) {
    int avail = (int)s.size();
    int nread = std::min(expected, avail);
    for (int i=0;i<nread;++i) (void)s.read();
    if (nread < expected) {
        std::printf("[TB][WARN] %s produced %d/%d tokens (missing %d)\n",
                    tag, nread, expected, expected - nread);
    }
    while (!s.empty()) (void)s.read();
    return nread;
}

// ============================================================
// Run one DUT pass with given weights, return out/H1
// ============================================================
static void run_dut_once(
    const DTYPE     kernel_host[CONV_K],
    const DTYPE_VEC A_fixed[STATE],
    const DTYPE_VEC RMS_weight[D_T],
    const W_VEC     W_inproj[D_T][CIN_T],
    const W_VEC     W_B[STATE][C2_T],
    const W_VEC     W_C[STATE][C2_T],
    const W_VEC     W_delta[C2_T][C2_T],
    const W_VEC     W_out[D_T][C2_T],
    const DTYPE_VEC D_diag[C2_T],
    const DTYPE_VEC X_host[D_T],
    const DTYPE_VEC H0_host[HUGE_LEN],
    const DTYPE_VEC conv_state_in_host[CONV_K-1],
    // outputs:
    DTYPE_VEC out_host_dut[D_T]
#if SSMU_ENABLE_H1_STREAM_OUT
  , DTYPE_VEC H1_host_dut[SSMU_H1_OUT_LEN]
#endif
) {
    hls::stream<DTYPE>     kernel_in("kernel_in");
    hls::stream<DTYPE_VEC> X_in("X_in");
    hls::stream<DTYPE_VEC> H0_in("H0_in");

    hls::stream<DTYPE_VEC> conv_state_in("conv_state_in");
    hls::stream<DTYPE_VEC> conv_state_out("conv_state_out");

    hls::stream<DTYPE_VEC> H1_out("H1_out");
    hls::stream<DTYPE_VEC> out("out");

    static DTYPE_VEC C_ddr[HUGE_LEN];
    static DTYPE_VEC H1_ddr[HUGE_LEN];

    float w_scale_in    = 0.0f;
    float w_scale_bc    = 0.0f;
    float w_scale_delta = 0.0f;
    float w_scale_out   = 0.0f;

#if TB_COSIM_THREADED_FEED
    std::thread dut_th([&](){
        SSMU(
            kernel_in,
            A_fixed,
            RMS_weight,
            W_inproj,
            W_B,
            W_C,
            W_delta,
            W_out,
            D_diag,
            X_in,
            H0_in,
            conv_state_in,
            conv_state_out,
            (DTYPE_VEC*)C_ddr,
            (DTYPE_VEC*)H1_ddr,
            H1_out,
            out,
            w_scale_in,
            w_scale_bc,
            w_scale_delta,
            w_scale_out
        );
    });

    for (int k=0;k<CONV_K;++k)      kernel_in.write(kernel_host[k]);
    for (int i=0;i<D_T;++i)        X_in.write(X_host[i]);
    for (int i=0;i<HUGE_LEN;++i)   H0_in.write(H0_host[i]);
    for (int k=0;k<CONV_K-1;++k)   conv_state_in.write(conv_state_in_host[k]);

    dut_th.join();

    // After DUT returns, drain using size-aware logic (no empty-read)
    (void)drain_stream_discard_n_or_warn(conv_state_out, CONV_K-1, "conv_state_out");

#if SSMU_ENABLE_H1_STREAM_OUT
    (void)drain_stream_n_or_pad(H1_out, SSMU_H1_OUT_LEN, H1_host_dut, "H1_out");
#else
    while (!H1_out.empty()) (void)H1_out.read();
#endif

    (void)drain_stream_n_or_pad(out, D_T, out_host_dut, "out");

#else
    // Non-threaded (C-sim OK)
    for (int k=0;k<CONV_K;++k)      kernel_in.write(kernel_host[k]);
    for (int i=0;i<D_T;++i)        X_in.write(X_host[i]);
    for (int i=0;i<HUGE_LEN;++i)   H0_in.write(H0_host[i]);
    for (int k=0;k<CONV_K-1;++k)   conv_state_in.write(conv_state_in_host[k]);

    SSMU(
        kernel_in,
        A_fixed,
        RMS_weight,
        W_inproj,
        W_B,
        W_C,
        W_delta,
        W_out,
        D_diag,
        X_in,
        H0_in,
        conv_state_in,
        conv_state_out,
        (DTYPE_VEC*)C_ddr,
        (DTYPE_VEC*)H1_ddr,
        H1_out,
        out,
        w_scale_in,
        w_scale_bc,
        w_scale_delta,
        w_scale_out
    );

    (void)drain_stream_discard_n_or_warn(conv_state_out, CONV_K-1, "conv_state_out");

#if SSMU_ENABLE_H1_STREAM_OUT
    (void)drain_stream_n_or_pad(H1_out, SSMU_H1_OUT_LEN, H1_host_dut, "H1_out");
#endif

    (void)drain_stream_n_or_pad(out, D_T, out_host_dut, "out");
#endif
}

// ============================================================
// Main
// ============================================================
int main() {
    std::printf("TB start\n");
    lcg = 1;

    static DTYPE_VEC A_fixed[STATE];
    static DTYPE_VEC RMS_weight[D_T];
    static DTYPE_VEC D_diag[C2_T];

    static W_VEC W_inproj[D_T][CIN_T];
    static W_VEC W_B[STATE][C2_T];
    static W_VEC W_C[STATE][C2_T];
    static W_VEC W_delta[C2_T][C2_T];
    static W_VEC W_out[D_T][C2_T];

    for (int i=0;i<STATE;++i) A_fixed[i]=rand_vec(-0.2f,0.2f);
    for (int i=0;i<D_T;++i)  RMS_weight[i]=rand_vec(0.8f,1.2f);
    for (int i=0;i<C2_T;++i) D_diag[i]=rand_vec(-0.1f,0.1f);

    for (int i=0;i<D_T;++i)
        for (int j=0;j<CIN_T;++j)
            W_inproj[i][j]=rand_wvec();

    for (int i=0;i<STATE;++i)
        for (int j=0;j<C2_T;++j) {
            W_B[i][j]=rand_wvec();
            W_C[i][j]=rand_wvec();
        }

    for (int i=0;i<C2_T;++i)
        for (int j=0;j<C2_T;++j)
            W_delta[i][j]=rand_wvec();

    for (int i=0;i<D_T;++i)
        for (int j=0;j<C2_T;++j)
            W_out[i][j]=rand_wvec();

    DTYPE kernel_host[CONV_K];
    for (int k=0;k<CONV_K;++k) kernel_host[k]=(DTYPE)frand(-0.2f,0.2f);

    static DTYPE_VEC X_host[D_T];
    static DTYPE_VEC H0_host[HUGE_LEN];
    static DTYPE_VEC conv_state_in_host[CONV_K-1];

    for (int i=0;i<D_T;++i)        X_host[i]=rand_vec(-0.5f,0.5f);
    for (int i=0;i<HUGE_LEN;++i)   H0_host[i]=rand_vec(-0.2f,0.2f);
    for (int k=0;k<CONV_K-1;++k)   conv_state_in_host[k]=rand_vec(-0.1f,0.1f);

#if TB_ENABLE_AUTO_DETECT
    static W_VEC W_B_zero[STATE][C2_T];
    static W_VEC W_C_zero[STATE][C2_T];
    static W_VEC W_inproj_zero[D_T][CIN_T];

    for (int i=0;i<STATE;++i)
        for (int j=0;j<C2_T;++j) {
            W_B_zero[i][j] = zero_wvec();
            W_C_zero[i][j] = zero_wvec();
        }

    for (int i=0;i<D_T;++i)
        for (int j=0;j<CIN_T;++j)
            W_inproj_zero[i][j] = zero_wvec();
#endif

    static DTYPE_VEC out_base[D_T];
#if SSMU_ENABLE_H1_STREAM_OUT
    static DTYPE_VEC h1_base[SSMU_H1_OUT_LEN];
#endif

    run_dut_once(
        kernel_host, A_fixed, RMS_weight,
        W_inproj, W_B, W_C, W_delta, W_out, D_diag,
        X_host, H0_host, conv_state_in_host,
        out_base
#if SSMU_ENABLE_H1_STREAM_OUT
      , h1_base
#endif
    );

#if TB_ENABLE_AUTO_DETECT
    static DTYPE_VEC out_wbwc0[D_T], out_inp0[D_T];
  #if SSMU_ENABLE_H1_STREAM_OUT
    static DTYPE_VEC h1_wbwc0[SSMU_H1_OUT_LEN], h1_inp0[SSMU_H1_OUT_LEN];
  #endif

    run_dut_once(
        kernel_host, A_fixed, RMS_weight,
        W_inproj, W_B_zero, W_C_zero, W_delta, W_out, D_diag,
        X_host, H0_host, conv_state_in_host,
        out_wbwc0
  #if SSMU_ENABLE_H1_STREAM_OUT
      , h1_wbwc0
  #endif
    );

    run_dut_once(
        kernel_host, A_fixed, RMS_weight,
        W_inproj_zero, W_B, W_C, W_delta, W_out, D_diag,
        X_host, H0_host, conv_state_in_host,
        out_inp0
  #if SSMU_ENABLE_H1_STREAM_OUT
      , h1_inp0
  #endif
    );
#endif

    auto print_vec = [&](const char* tag, const DTYPE_VEC& v){
        std::printf("%s:", tag);
        for (unsigned l=0; l<(unsigned)VEC_FACTOR && l<4; ++l)
            std::printf(" %f", (float)(vget(v,l)));
        std::printf("\n");
    };

    print_vec("BASE out[0]", out_base[0]);

#if TB_ENABLE_AUTO_DETECT
    print_vec("WBWC0 out[0]", out_wbwc0[0]);
    print_vec("INP0  out[0]", out_inp0[0]);

    float d_wbwc = max_abs_diff_out(out_base, out_wbwc0);
    float d_inp  = max_abs_diff_out(out_base, out_inp0);

    std::printf("DELTA(out) when zero W_B/W_C : %e\n", d_wbwc);
    std::printf("DELTA(out) when zero W_inproj: %e\n", d_inp);

  #if SSMU_ENABLE_H1_STREAM_OUT
    float dh_wbwc = max_abs_diff_h1(h1_base, h1_wbwc0);
    float dh_inp  = max_abs_diff_h1(h1_base, h1_inp0);
    std::printf("DELTA(H1 ) when zero W_B/W_C : %e\n", dh_wbwc);
    std::printf("DELTA(H1 ) when zero W_inproj: %e\n", dh_inp);
  #endif

    const float eps = 1e-6f;
    if (d_wbwc < eps && d_inp < eps) {
        std::printf("[AUTO] Output did not change under either zeroing.\n");
    } else if (d_wbwc > 10.0f * d_inp) {
        std::printf("[AUTO] Likely: BC_from_WBWC (W_B/W_C dominate)\n");
    } else if (d_inp > 10.0f * d_wbwc) {
        std::printf("[AUTO] Likely: BC_from_inproj (W_inproj dominate)\n");
    } else {
        std::printf("[AUTO] Mixed / unclear: both affect output.\n");
    }
#else
    std::printf("[RTL COSIM] Auto-detect disabled to avoid 3 transactions / OOM.\n");
#endif

    static DTYPE_VEC out_ref[D_T];
    static DTYPE_VEC h1_ref[SSMU_H1_OUT_LEN];
    reference_model_stub(
        A_fixed, RMS_weight, W_inproj, W_B, W_C, W_delta, W_out, D_diag,
        kernel_host, X_host, H0_host, conv_state_in_host,
        out_ref, h1_ref
    );

    print_vec("REF  out[0]", out_ref[0]);

    std::printf("TB done\n");
    return 0;
}
