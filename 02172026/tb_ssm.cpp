// tb_ssmu.cpp  (BIN-DRIVEN + GOLDEN-COMPARE / COSIM-safe)
// - Reads *.raw.bin (int16) from TB_BINS_DIR (override), otherwise auto-locate bins_raw (relative)
// - Converts int16 (Q4.12) -> DTYPE (assume ap_fixed<16,4> with FRAC_BITS=12)
// - For vec tokens: supports either
//     (A) vec layout: tokens*VEC_FACTOR elems  (preferred)
//     (B) scalar layout: tokens elems -> broadcast to all lanes
// - Runs DUT once (threaded feed), captures OUT + H1_out, compares vs out_ref/h1_ref
//
// Files required in bins_raw:
//   A_fixed_f32.raw.bin          (NOTE: must be STATE_V tokens = SSMU_STATE_T)
//   RMS_weight_f32.raw.bin
//   D_diag_f32.raw.bin
//   kernel_in_f32.raw.bin
//   x_in_f32.raw.bin
//   h0_in_f32.raw.bin            (NOTE: must be H0_LEN tokens = SSMU_H1_OUT_LEN = STATE_V*C2_T)
//   conv_state_in_f32.raw.bin
//   W_inproj_f32.raw.bin
//   W_delta_f32.raw.bin
//   W_out_f32.raw.bin
//   out_ref_f32.raw.bin
//   h1_ref_f32.raw.bin           (NOTE: must be SSMU_H1_OUT_LEN tokens)
//
// -----------------------------------------------------------------------------
// IMPORTANT ALIGNMENT WITH YOUR ssm.cpp (STATE_V = SSMU_STATE_T):
//   - A_fixed tokens:  STATE_V
//   - H0_in tokens:    SSMU_H1_OUT_LEN = STATE_V * C2_T
//   - H1_out tokens:   SSMU_H1_OUT_LEN
// -----------------------------------------------------------------------------

#include "SSMU.h"

#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <thread>
#include <string>
#include <vector>

// ============================================================
// Shapes from SSMU.h
// ============================================================
static const int D_T     = SSMU_D_T;
static const int C2_T    = SSMU_C2_T;
static const int CCONV_T = SSMU_CCONV_T;
static const int CH_T    = SSMU_CH_T;
static const int CIN_T   = SSMU_CIN_T;
static const int CONV_K  = SSMU_K;

#ifndef SSMU_STATE_T
#define SSMU_STATE_T (SSMU_STATE / VEC_FACTOR)
#endif

static const int STATE_SCALAR = SSMU_STATE;     // scalar state count (for reference only)
static const int STATE_V      = SSMU_STATE_T;   // vectorized state count (THIS is what DUT uses)

#ifndef SSMU_ENABLE_H1_STREAM_OUT
#define SSMU_ENABLE_H1_STREAM_OUT 1
#endif

#ifndef SSMU_H1_OUT_LEN
#define SSMU_H1_OUT_LEN (SSMU_STATE_T * SSMU_C2_T)
#endif

// H0 tokens must match DUT contract
static const int H0_LEN = SSMU_H1_OUT_LEN;

#ifndef VEC_FACTOR
#error "VEC_FACTOR must be defined in SSMU.h"
#endif

// ============================================================
// If you packed with FRAC_BITS=12 (ap_fixed<16,4>), keep this:
// ============================================================
#ifndef TB_FRAC_BITS
#define TB_FRAC_BITS 12
#endif

// ============================================================
// vget/vset fallback (if header doesn't provide)
// ============================================================
#ifndef SSMU_HAVE_VGET_VSET
template<typename V>
static inline auto vget(const V& v, unsigned idx) -> decltype(v[idx]) { return v[idx]; }
template<typename V, typename T>
static inline void vset(V& v, unsigned idx, const T& val) { v[idx] = (decltype(v[idx]))val; }
#endif

static inline DTYPE from_i16(int16_t q) {
    float f = (float)q / (float)(1 << TB_FRAC_BITS);
    return (DTYPE)f;
}

static inline DTYPE_VEC make_vec_zero() {
    DTYPE_VEC v;
    for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) vset(v, l, (DTYPE)0);
    return v;
}

static void print_vec4(const char* tag, const DTYPE_VEC& v) {
    std::printf("%s:", tag);
    for (unsigned l=0; l<(unsigned)VEC_FACTOR && l<4; ++l) {
        std::printf(" %f", (float)vget(v,l));
    }
    std::printf("\n");
}

// ============================================================
// Path + file utils
// ============================================================
static std::string join_path(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (a.back() == '/') return a + b;
    return a + "/" + b;
}

static bool read_file_i16(const std::string& path, std::vector<int16_t>& out) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;

    std::fseek(fp, 0, SEEK_END);
    long sz = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);

    if (sz < 0) { std::fclose(fp); return false; }
    if ((sz % (long)sizeof(int16_t)) != 0) {
        std::printf("[TB][FATAL] file size not multiple of int16: %s size=%ld\n", path.c_str(), sz);
        std::fclose(fp);
        return false;
    }

    size_t n = (size_t)sz / sizeof(int16_t);
    out.resize(n);
    size_t r = std::fread(out.data(), sizeof(int16_t), n, fp);
    std::fclose(fp);
    return r == n;
}

// ============================================================
// bins_raw auto-locator (Method A)
// - If TB_BINS_DIR is set: use it
// - Else try common relative paths: bins_raw, ../bins_raw, ../../bins_raw ...
// - Validate by checking existence of A_fixed_f32.raw.bin
// ============================================================
static std::string resolve_bins_dir() {
    if (const char* env = std::getenv("TB_BINS_DIR")) {
        if (env[0]) return std::string(env);
    }

    const char* cands[] = {
        "bins_raw",
        "../bins_raw",
        "../../bins_raw",
        "../../../bins_raw",
        "../../../../bins_raw"
    };

    for (auto* c : cands) {
        std::vector<int16_t> tmp;
        const std::string probe = join_path(std::string(c), "A_fixed_f32.raw.bin");
        if (read_file_i16(probe, tmp)) {
            return std::string(c);
        }
    }

    return std::string("bins_raw");
}

// Read vec tokens into DTYPE_VEC[tokens].
// Accept either:
//  - vec layout: tokens*VEC_FACTOR elems
//  - scalar layout: tokens elems (broadcast to all lanes)
static bool load_vec_tokens(
    const std::string& path,
    int tokens,
    DTYPE_VEC* dst,
    const char* tag
) {
    std::vector<int16_t> buf;
    if (!read_file_i16(path, buf)) {
        std::printf("[TB][FATAL] cannot open: %s\n", path.c_str());
        return false;
    }

    const long got = (long)buf.size();
    const long exp_vec = (long)tokens * (long)VEC_FACTOR;
    const long exp_sca = (long)tokens;

    if (got == exp_vec) {
        long idx = 0;
        for (int t=0; t<tokens; ++t) {
            DTYPE_VEC v;
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
                vset(v, l, from_i16(buf[idx++]));
            }
            dst[t] = v;
        }
        std::printf("[TB] %s loaded (vec) elems=%ld tokens=%d\n", tag, got, tokens);
        return true;
    }

    if (got == exp_sca) {
        for (int t=0; t<tokens; ++t) {
            DTYPE x = from_i16(buf[t]);
            DTYPE_VEC v;
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) vset(v, l, x);
            dst[t] = v;
        }
        std::printf("[TB][WARN] %s loaded (scalar->broadcast) elems=%ld tokens=%d (expected vec=%ld)\n",
                    tag, got, tokens, exp_vec);
        return true;
    }

    std::printf("[TB][FATAL] elem count mismatch: %s got=%ld expected(vec)=%ld or scalar=%ld\n",
                path.c_str(), got, exp_vec, exp_sca);
    return false;
}

// Read scalar tokens into DTYPE[tokens] (kernel_in)
static bool load_scalar_tokens(
    const std::string& path,
    int tokens,
    DTYPE* dst,
    const char* tag
) {
    std::vector<int16_t> buf;
    if (!read_file_i16(path, buf)) {
        std::printf("[TB][FATAL] cannot open: %s\n", path.c_str());
        return false;
    }
    long got = (long)buf.size();
    if (got != tokens) {
        std::printf("[TB][FATAL] elem count mismatch: %s got=%ld expected=%d\n",
                    path.c_str(), got, tokens);
        return false;
    }
    for (int i=0;i<tokens;++i) dst[i] = from_i16(buf[i]);
    std::printf("[TB] %s loaded (scalar) elems=%ld\n", tag, got);
    return true;
}

// Load W_VEC matrix stored as vec tokens (each element is W_VEC lane vector)
// Expect elems = ROWS*COLS*VEC_FACTOR OR scalar ROWS*COLS (broadcast lanes).
template<int ROWS, int COLS>
static bool load_w_mat(
    const std::string& path,
    W_VEC dst[ROWS][COLS],
    const char* tag
) {
    std::vector<int16_t> buf;
    if (!read_file_i16(path, buf)) {
        std::printf("[TB][FATAL] cannot open: %s\n", path.c_str());
        return false;
    }
    const long got = (long)buf.size();
    const long exp_vec = (long)ROWS * (long)COLS * (long)VEC_FACTOR;
    const long exp_sca = (long)ROWS * (long)COLS;

    if (got == exp_vec) {
        long idx = 0;
        for (int r=0;r<ROWS;++r) for (int c=0;c<COLS;++c) {
            W_VEC w;
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
#if SSMU_USE_INT8
                // int8 mode requires different bin packing; this TB assumes SSMU_USE_INT8=0.
                vset(w, l, (ap_int<8>)0);
#else
                vset(w, l, (DTYPE)from_i16(buf[idx++]));
#endif
            }
            dst[r][c] = w;
        }
        std::printf("[TB] %s loaded (vec) elems=%ld\n", tag, got);
        return true;
    }

    if (got == exp_sca) {
        long idx = 0;
        for (int r=0;r<ROWS;++r) for (int c=0;c<COLS;++c) {
            DTYPE x = from_i16(buf[idx++]);
            W_VEC w;
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
#if SSMU_USE_INT8
                vset(w, l, (ap_int<8>)0);
#else
                vset(w, l, x);
#endif
            }
            dst[r][c] = w;
        }
        std::printf("[TB][WARN] %s loaded (scalar->broadcast) elems=%ld (expected vec=%ld)\n",
                    tag, got, exp_vec);
        return true;
    }

    std::printf("[TB][FATAL] elem count mismatch: %s got=%ld expected(vec)=%ld or scalar=%ld\n",
                path.c_str(), got, exp_vec, exp_sca);
    return false;
}

// ============================================================
// Stream drain
// ============================================================
template<typename T>
static void drain_all(hls::stream<T>& s) { while (!s.empty()) (void)s.read(); }

// ============================================================
// Run DUT once, capture out + h1
// ============================================================
static int run_dut_once(
    const DTYPE     kernel_host[CONV_K],
    const DTYPE_VEC A_fixed[STATE_V],
    const DTYPE_VEC RMS_weight[D_T],
    const W_VEC     W_inproj[D_T][CIN_T],
    const W_VEC     W_delta[C2_T][C2_T],
    const W_VEC     W_out[D_T][C2_T],
    const DTYPE_VEC D_diag[C2_T],
    const DTYPE_VEC X_host[D_T],
    const DTYPE_VEC H0_host[H0_LEN],
    const DTYPE_VEC conv_state_in_host[CONV_K-1],
    // outputs:
    DTYPE_VEC out_host_dut[D_T],
    DTYPE_VEC h1_host_dut[SSMU_H1_OUT_LEN],
    int& h1_produced
) {
    hls::stream<DTYPE>      kernel_in("kernel_in");
    hls::stream<DTYPE_VEC>  X_in("X_in");
    hls::stream<DTYPE_VEC>  H0_in("H0_in");
    hls::stream<DTYPE_VEC>  conv_state_in("conv_state_in");
    hls::stream<DTYPE_VEC>  conv_state_out("conv_state_out");
    hls::stream<DTYPE_VEC>  H1_out("H1_out");
    hls::stream<DTYPE_VEC>  out("out");

    static DTYPE_VEC C_ddr[H0_LEN];
    static DTYPE_VEC H1_ddr[H0_LEN];

    float w_scale_in    = 0.0f;
    float w_scale_delta = 0.0f;
    float w_scale_out   = 0.0f;

    std::thread dut_th([&](){
        SSMU(
            kernel_in,
            A_fixed,
            RMS_weight,
            W_inproj,
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
            w_scale_delta,
            w_scale_out
        );
    });

    // Feed after DUT starts (cosim-safe style)
    for (int k=0;k<CONV_K;++k)      kernel_in.write(kernel_host[k]);
    for (int i=0;i<D_T;++i)        X_in.write(X_host[i]);
    for (int i=0;i<H0_LEN;++i)     H0_in.write(H0_host[i]);
    for (int k=0;k<CONV_K-1;++k)   conv_state_in.write(conv_state_in_host[k]);

    dut_th.join();

    // conv_state_out: drain all
    drain_all(conv_state_out);

    // capture H1_out up to cap
    h1_produced = (int)H1_out.size();
    int h1_read = 0;
    while (!H1_out.empty() && h1_read < (int)SSMU_H1_OUT_LEN) {
        h1_host_dut[h1_read++] = H1_out.read();
    }
    for (int i=h1_read; i<(int)SSMU_H1_OUT_LEN; ++i) h1_host_dut[i] = make_vec_zero();
    drain_all(H1_out);

    // capture out up to D_T
    int out_read = 0;
    while (!out.empty() && out_read < D_T) {
        out_host_dut[out_read++] = out.read();
    }
    for (int i=out_read; i<D_T; ++i) out_host_dut[i] = make_vec_zero();
    drain_all(out);

    return 0;
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

static float max_abs_diff_vec_tokens(const DTYPE_VEC* a, const DTYPE_VEC* b, int tokens) {
    float m = 0.0f;
    for (int i=0;i<tokens;++i) m = std::max(m, vec_max_abs_diff(a[i], b[i]));
    return m;
}

// ============================================================
// Env float reader
// ============================================================
static float read_env_float(const char* name, float defv) {
    const char* s = std::getenv(name);
    if (!s || !s[0]) return defv;
    return (float)std::atof(s);
}

// ============================================================
// Main
// ============================================================
int main() {
    std::string bins_dir = resolve_bins_dir();

    const float tol_out = read_env_float("TB_TOL_OUT", 0.05f);
    const float tol_h1  = read_env_float("TB_TOL_H1",  0.06f);

    std::printf("[TB] Using bins dir: %s\n", bins_dir.c_str());
    std::printf("[TB] STATE_SCALAR=%d  STATE_V(STATE_T)=%d  D_T=%d  C2_T=%d  CIN_T=%d  VEC_FACTOR=%d\n",
                (int)STATE_SCALAR, (int)STATE_V, (int)D_T, (int)C2_T, (int)CIN_T, (int)VEC_FACTOR);
    std::printf("[TB] H0_LEN=%d  H1_LEN=%d\n", (int)H0_LEN, (int)SSMU_H1_OUT_LEN);
    std::printf("[TB] TOL_OUT=%g  TOL_H1=%g  (set TB_TOL_OUT/TB_TOL_H1 to override)\n", tol_out, tol_h1);

    // host arrays
    static DTYPE     kernel_host[CONV_K];
    static DTYPE_VEC X_host[D_T];
    static DTYPE_VEC H0_host[H0_LEN];
    static DTYPE_VEC conv_state_in_host[CONV_K-1];

    static DTYPE_VEC A_fixed[STATE_V];
    static DTYPE_VEC RMS_weight[D_T];
    static DTYPE_VEC D_diag[C2_T];

    static W_VEC W_inproj[D_T][CIN_T];
    static W_VEC W_delta[C2_T][C2_T];
    static W_VEC W_out[D_T][C2_T];

    static DTYPE_VEC out_ref[D_T];
    static DTYPE_VEC h1_ref[SSMU_H1_OUT_LEN];

    // paths
    auto P = [&](const char* fn){ return join_path(bins_dir, std::string(fn)); };

    // ---- load inputs/weights ----
    if (!load_vec_tokens(P("A_fixed_f32.raw.bin"),          STATE_V, A_fixed,    "A_fixed")) return 1;
    if (!load_vec_tokens(P("RMS_weight_f32.raw.bin"),       D_T,     RMS_weight, "RMS_weight")) return 1;
    if (!load_vec_tokens(P("D_diag_f32.raw.bin"),           C2_T,    D_diag,     "D_diag")) return 1;

    if (!load_scalar_tokens(P("kernel_in_f32.raw.bin"),     CONV_K,  kernel_host,"kernel_in")) return 1;
    if (!load_vec_tokens(P("x_in_f32.raw.bin"),             D_T,     X_host,     "x_in")) return 1;
    if (!load_vec_tokens(P("h0_in_f32.raw.bin"),            H0_LEN,  H0_host,    "h0_in")) return 1;
    if (!load_vec_tokens(P("conv_state_in_f32.raw.bin"),    CONV_K-1,conv_state_in_host,"conv_state_in")) return 1;

    if (!load_w_mat<D_T, CIN_T>(P("W_inproj_f32.raw.bin"), W_inproj, "W_inproj")) return 1;
    if (!load_w_mat<C2_T, C2_T>(P("W_delta_f32.raw.bin"),  W_delta,  "W_delta")) return 1;
    if (!load_w_mat<D_T, C2_T>(P("W_out_f32.raw.bin"),     W_out,    "W_out")) return 1;

    // ---- load golden refs ----
    if (!load_vec_tokens(P("out_ref_f32.raw.bin"),          D_T,     out_ref,    "out_ref")) return 1;
    if (!load_vec_tokens(P("h1_ref_f32.raw.bin"),           SSMU_H1_OUT_LEN, h1_ref, "h1_ref")) return 1;

    // ---- run dut ----
    static DTYPE_VEC out_dut[D_T];
    static DTYPE_VEC h1_dut[SSMU_H1_OUT_LEN];
    int h1_produced = 0;

    int rc = run_dut_once(
        kernel_host, A_fixed, RMS_weight,
        W_inproj, W_delta, W_out, D_diag,
        X_host, H0_host, conv_state_in_host,
        out_dut, h1_dut, h1_produced
    );
    if (rc) return rc;

    // ---- compare ----
    float diff_out = max_abs_diff_vec_tokens(out_dut, out_ref, D_T);

#if SSMU_ENABLE_H1_STREAM_OUT
    float diff_h1  = max_abs_diff_vec_tokens(h1_dut, h1_ref, SSMU_H1_OUT_LEN);
#else
    float diff_h1 = 0.0f;
#endif

    print_vec4("DUT out[0]", out_dut[0]);
    print_vec4("REF out[0]", out_ref[0]);

#if SSMU_ENABLE_H1_STREAM_OUT
    std::printf("[TB] H1_out produced (stream.size snapshot) = %d, compare_len=%d\n",
                h1_produced, (int)SSMU_H1_OUT_LEN);
    print_vec4("DUT h1[0]", h1_dut[0]);
    print_vec4("REF h1[0]", h1_ref[0]);
#endif

    std::printf("[TB] max_abs_diff(out) = %e\n", diff_out);
#if SSMU_ENABLE_H1_STREAM_OUT
    std::printf("[TB] max_abs_diff(h1 ) = %e\n", diff_h1);
#endif

    bool pass = (diff_out <= tol_out)
#if SSMU_ENABLE_H1_STREAM_OUT
             && (diff_h1  <= tol_h1)
#endif
             ;

    if (pass) {
        std::printf("[TB][PASS] Golden match within tolerance.\n");
        return 0;
    } else {
        std::printf("[TB][FAIL] Golden mismatch.\n");
        std::printf("            (set TB_TOL_OUT / TB_TOL_H1 if using approx math)\n");
        return 2;
    }
}
