#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    int d_model;
    int expand;
    int d_inner;
    int d_state;
    int nheads;
    int headdim;
    int conv_kernel;
    bool rmsnorm;
    bool norm_before_gate;
} Mamba2Config;

typedef struct {
    Mamba2Config cfg;

    // Packed input projection: [z, xBC, dt] where xBC = [x, B, C]
    // w_in shape = [d_model, 2*d_inner + 2*d_state + nheads]
    float *w_in;
    float *b_in;

    // Depthwise conv over xBC channels.
    float *conv_w;    // [d_inner + 2*d_state, conv_kernel]
    float *conv_b;    // [d_inner + 2*d_state]

    // Head-wise SSM parameters.
    float *a_log;     // [nheads]
    float *d_skip;    // [nheads]
    float *dt_bias;   // [nheads]
    float *norm_w;    // [d_inner], RMSNormGated scale

    float *w_out;     // [d_inner, d_model]
    float *b_out;     // [d_model]

    // Runtime state.
    float *ssm_state; // [d_inner, d_state]
    float *conv_buf;  // [conv_kernel, d_inner + 2*d_state]
    int conv_pos;
} Mamba2Layer;

static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

static inline float softplus(float x) {
    if (x > 20.0f) {
        return x;
    }
    return log1pf(expf(x));
}

static inline float rand_uniform(float lo, float hi) {
    float r = (float)rand() / (float)RAND_MAX;
    return lo + (hi - lo) * r;
}

static bool alloc_float(float **ptr, size_t count) {
    *ptr = (float *)malloc(count * sizeof(float));
    return *ptr != NULL;
}

static bool alloc_zero_float(float **ptr, size_t count) {
    *ptr = (float *)calloc(count, sizeof(float));
    return *ptr != NULL;
}

static void init_random(float *ptr, size_t n, float scale) {
    for (size_t i = 0; i < n; i++) {
        ptr[i] = rand_uniform(-scale, scale);
    }
}

static Mamba2Config mamba2_config_27b(void) {
    Mamba2Config cfg;
    cfg.d_model = 2560;
    cfg.expand = 2;
    cfg.d_inner = cfg.d_model * cfg.expand; // 5120
    cfg.d_state = 128;
    cfg.headdim = 64;
    cfg.nheads = cfg.d_inner / cfg.headdim; // 80
    cfg.conv_kernel = 4;
    cfg.rmsnorm = true;
    cfg.norm_before_gate = false;
    return cfg;
}

static int in_proj_width(const Mamba2Config *cfg) {
    return 2 * cfg->d_inner + 2 * cfg->d_state + cfg->nheads;
}

static int xbc_width(const Mamba2Config *cfg) {
    return cfg->d_inner + 2 * cfg->d_state;
}

static bool mamba2_layer_init(Mamba2Layer *layer, Mamba2Config cfg, uint32_t seed) {
    memset(layer, 0, sizeof(*layer));
    layer->cfg = cfg;

    if (cfg.d_inner % cfg.headdim != 0 || cfg.nheads * cfg.headdim != cfg.d_inner) {
        fprintf(stderr, "Invalid head config: d_inner=%d, nheads=%d, headdim=%d\n", cfg.d_inner,
                cfg.nheads, cfg.headdim);
        return false;
    }

    srand(seed);

    const int d_model = cfg.d_model;
    const int d_inner = cfg.d_inner;
    const int d_state = cfg.d_state;
    const int nheads = cfg.nheads;
    const int k = cfg.conv_kernel;
    const int w_in_cols = in_proj_width(&cfg);
    const int xbc_dim = xbc_width(&cfg);

    size_t in_proj = (size_t)d_model * (size_t)w_in_cols;
    size_t out_proj = (size_t)d_inner * (size_t)d_model;
    size_t ssm_state_size = (size_t)d_inner * (size_t)d_state;
    size_t conv_params = (size_t)xbc_dim * (size_t)k;

    if (!alloc_float(&layer->w_in, in_proj) || !alloc_float(&layer->b_in, (size_t)w_in_cols) ||
        !alloc_float(&layer->conv_w, conv_params) || !alloc_float(&layer->conv_b, (size_t)xbc_dim) ||
        !alloc_float(&layer->a_log, (size_t)nheads) || !alloc_float(&layer->d_skip, (size_t)nheads) ||
        !alloc_float(&layer->dt_bias, (size_t)nheads) || !alloc_float(&layer->norm_w, (size_t)d_inner) ||
        !alloc_float(&layer->w_out, out_proj) ||
        !alloc_float(&layer->b_out, (size_t)d_model) ||
        !alloc_zero_float(&layer->ssm_state, ssm_state_size) ||
        !alloc_zero_float(&layer->conv_buf, (size_t)k * (size_t)xbc_dim)) {
        return false;
    }

    layer->conv_pos = 0;

    init_random(layer->w_in, in_proj, 0.02f);
    init_random(layer->b_in, (size_t)w_in_cols, 0.01f);

    init_random(layer->conv_w, conv_params, 0.03f);
    init_random(layer->conv_b, (size_t)xbc_dim, 0.01f);

    for (int h = 0; h < nheads; h++) {
        layer->a_log[h] = rand_uniform(-1.5f, 0.5f);
    }

    init_random(layer->d_skip, (size_t)nheads, 0.05f);

    for (int h = 0; h < nheads; h++) {
        layer->dt_bias[h] = rand_uniform(-2.0f, -0.5f);
    }
    for (int i = 0; i < d_inner; i++) {
        layer->norm_w[i] = 1.0f;
    }

    init_random(layer->w_out, out_proj, 0.02f);
    init_random(layer->b_out, (size_t)d_model, 0.01f);

    return true;
}

static void mamba2_layer_free(Mamba2Layer *layer) {
    free(layer->w_in);
    free(layer->b_in);
    free(layer->conv_w);
    free(layer->conv_b);
    free(layer->a_log);
    free(layer->d_skip);
    free(layer->dt_bias);
    free(layer->norm_w);
    free(layer->w_out);
    free(layer->b_out);
    free(layer->ssm_state);
    free(layer->conv_buf);
    memset(layer, 0, sizeof(*layer));
}

static void rmsnorm_gated(float *y, const float *z, int n, const float *w, bool norm_before_gate) {
    const float eps = 1e-5f;
    double sq = 0.0;

    if (norm_before_gate) {
        for (int i = 0; i < n; i++) {
            sq += (double)y[i] * (double)y[i];
        }
        float inv_rms = 1.0f / sqrtf((float)(sq / (double)n) + eps);
        for (int i = 0; i < n; i++) {
            y[i] = (y[i] * inv_rms * w[i]) * silu(z[i]);
        }
        return;
    }

    for (int i = 0; i < n; i++) {
        y[i] *= silu(z[i]);
        sq += (double)y[i] * (double)y[i];
    }
    float inv_rms = 1.0f / sqrtf((float)(sq / (double)n) + eps);
    for (int i = 0; i < n; i++) {
        y[i] = y[i] * inv_rms * w[i];
    }
}

static void mamba2_layer_reset_state(Mamba2Layer *layer) {
    const int d_inner = layer->cfg.d_inner;
    const int d_state = layer->cfg.d_state;
    const int k = layer->cfg.conv_kernel;
    const int xbc_dim = xbc_width(&layer->cfg);

    memset(layer->ssm_state, 0, (size_t)d_inner * (size_t)d_state * sizeof(float));
    memset(layer->conv_buf, 0, (size_t)k * (size_t)xbc_dim * sizeof(float));
    layer->conv_pos = 0;
}

// x: [seq_len, d_model], y: [seq_len, d_model]
static void mamba2_layer_forward(Mamba2Layer *layer, const float *x, int seq_len, float *y) {
    const int d_model = layer->cfg.d_model;
    const int d_inner = layer->cfg.d_inner;
    const int d_state = layer->cfg.d_state;
    const int headdim = layer->cfg.headdim;
    const int k = layer->cfg.conv_kernel;
    const int w_in_cols = in_proj_width(&layer->cfg);
    const int xbc_dim = xbc_width(&layer->cfg);

    const int off_z = 0;
    const int off_xbc = off_z + d_inner;
    const int off_dt = off_xbc + xbc_dim;

    float *proj = (float *)malloc((size_t)w_in_cols * sizeof(float));
    float *conv_xbc = (float *)malloc((size_t)xbc_dim * sizeof(float));
    float *z_tok = (float *)malloc((size_t)d_inner * sizeof(float));
    float *inner_out = (float *)malloc((size_t)d_inner * sizeof(float));

    if (!proj || !conv_xbc || !z_tok || !inner_out) {
        fprintf(stderr, "Allocation failure in forward pass.\n");
        free(proj);
        free(conv_xbc);
        free(z_tok);
        free(inner_out);
        return;
    }

    for (int t = 0; t < seq_len; t++) {
        const float *x_t = x + (size_t)t * (size_t)d_model;
        float *y_t = y + (size_t)t * (size_t)d_model;

        for (int j = 0; j < w_in_cols; j++) {
            proj[j] = layer->b_in[j];
        }

        for (int i = 0; i < d_model; i++) {
            float xi = x_t[i];
            const float *wrow = layer->w_in + (size_t)i * (size_t)w_in_cols;
            for (int j = 0; j < w_in_cols; j++) {
                proj[j] += xi * wrow[j];
            }
        }

        // Conv over xBC branch.
        float *buf_pos = layer->conv_buf + (size_t)layer->conv_pos * (size_t)xbc_dim;
        for (int i = 0; i < xbc_dim; i++) {
            buf_pos[i] = proj[off_xbc + i];
        }

        for (int i = 0; i < xbc_dim; i++) {
            float acc = layer->conv_b[i];
            const float *w = layer->conv_w + (size_t)i * (size_t)k;
            for (int off = 0; off < k; off++) {
                int idx = layer->conv_pos - off;
                while (idx < 0) {
                    idx += k;
                }
                idx %= k;
                const float *buf = layer->conv_buf + (size_t)idx * (size_t)xbc_dim;
                acc += w[off] * buf[i];
            }
            // Official Mamba2 applies activation to the conv branch before split.
            conv_xbc[i] = silu(acc);
        }

        layer->conv_pos = (layer->conv_pos + 1) % k;

        // Split activated conv branch to x, B, C.
        const float *x_tok = conv_xbc;
        const float *b_tok = conv_xbc + d_inner;
        const float *c_tok = conv_xbc + d_inner + d_state;
        const float *dt_tok = proj + off_dt; // [nheads]

        for (int i = 0; i < d_inner; i++) {
            int h = i / headdim;

            z_tok[i] = proj[off_z + i];
            float u = x_tok[i];

            float dt = softplus(dt_tok[h] + layer->dt_bias[h]) + 1e-4f;
            float a = -expf(layer->a_log[h]);
            float decay = expf(dt * a);

            float ssm_acc = 0.0f;
            size_t base = (size_t)i * (size_t)d_state;
            for (int n = 0; n < d_state; n++) {
                size_t idx = base + (size_t)n;
                float prev = layer->ssm_state[idx];

                float b = b_tok[n];
                float c = c_tok[n];

                float s_new = decay * prev + dt * b * u;
                layer->ssm_state[idx] = s_new;
                ssm_acc += c * s_new;
            }

            inner_out[i] = ssm_acc + layer->d_skip[h] * u;
        }

        if (layer->cfg.rmsnorm) {
            rmsnorm_gated(inner_out, z_tok, d_inner, layer->norm_w, layer->cfg.norm_before_gate);
        } else {
            for (int i = 0; i < d_inner; i++) {
                inner_out[i] *= silu(z_tok[i]);
            }
        }

        for (int j = 0; j < d_model; j++) {
            y_t[j] = layer->b_out[j];
        }

        for (int i = 0; i < d_inner; i++) {
            float yi = inner_out[i];
            const float *wrow = layer->w_out + (size_t)i * (size_t)d_model;
            for (int j = 0; j < d_model; j++) {
                y_t[j] += yi * wrow[j];
            }
        }
    }

    free(proj);
    free(conv_xbc);
    free(z_tok);
    free(inner_out);
}

// Software reference top function using externally provided parameters.
// Returns true on success, false on allocation/config failure.
bool mamba2_software_top(
    const Mamba2Config *cfg,
    const float *x, int seq_len, float *y,
    const float *w_in, const float *b_in,
    const float *conv_w, const float *conv_b,
    const float *a_log, const float *d_skip, const float *dt_bias,
    const float *norm_w,
    const float *w_out, const float *b_out) {
    if (!cfg || !x || !y || !w_in || !b_in || !conv_w || !conv_b ||
        !a_log || !d_skip || !dt_bias || !norm_w || !w_out || !b_out || seq_len <= 0) {
        return false;
    }

    Mamba2Layer layer;
    memset(&layer, 0, sizeof(layer));
    layer.cfg = *cfg;

    if (cfg->d_inner % cfg->headdim != 0 || cfg->nheads * cfg->headdim != cfg->d_inner) {
        return false;
    }

    layer.w_in = (float *)w_in;
    layer.b_in = (float *)b_in;
    layer.conv_w = (float *)conv_w;
    layer.conv_b = (float *)conv_b;
    layer.a_log = (float *)a_log;
    layer.d_skip = (float *)d_skip;
    layer.dt_bias = (float *)dt_bias;
    layer.norm_w = (float *)norm_w;
    layer.w_out = (float *)w_out;
    layer.b_out = (float *)b_out;

    const int xbc_dim = xbc_width(cfg);
    const int k = cfg->conv_kernel;
    const size_t ssm_state_size = (size_t)cfg->d_inner * (size_t)cfg->d_state;
    const size_t conv_buf_size = (size_t)k * (size_t)xbc_dim;

    if (!alloc_zero_float(&layer.ssm_state, ssm_state_size) ||
        !alloc_zero_float(&layer.conv_buf, conv_buf_size)) {
        free(layer.ssm_state);
        free(layer.conv_buf);
        return false;
    }

    mamba2_layer_reset_state(&layer);
    mamba2_layer_forward(&layer, x, seq_len, y);

    free(layer.ssm_state);
    free(layer.conv_buf);
    return true;
}

static bool all_finite(const float *v, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(v[i])) {
            return false;
        }
    }
    return true;
}

static float mean_abs(const float *v, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; i++) {
        s += fabs((double)v[i]);
    }
    return (float)(s / (double)n);
}

#ifndef MAMBA_SOFTWARE_NO_MAIN
int main(void) {
    Mamba2Config cfg = mamba2_config_27b();
    Mamba2Layer layer;
    const uint32_t weight_seed = 12345u;
    const uint32_t input_seed = 42u;

    printf("Initializing one Mamba2-style layer (2.7B dimensions): d_model=%d, d_inner=%d, d_state=%d, nheads=%d, headdim=%d, conv=%d\n",
           cfg.d_model, cfg.d_inner, cfg.d_state, cfg.nheads, cfg.headdim, cfg.conv_kernel);
    printf("w_in shape = [%d, %d] = [d_model, 2*d_inner + 2*d_state + nheads]\n",
           cfg.d_model, in_proj_width(&cfg));

    if (!mamba2_layer_init(&layer, cfg, weight_seed)) {
        fprintf(stderr, "Failed to initialize layer parameters.\n");
        return 1;
    }

    const int seq_len = 2;
    size_t n_in = (size_t)seq_len * (size_t)cfg.d_model;

    float *x = (float *)malloc(n_in * sizeof(float));
    float *y = (float *)malloc(n_in * sizeof(float));

    if (!x || !y) {
        fprintf(stderr, "Failed to allocate test tensors.\n");
        free(x);
        free(y);
        mamba2_layer_free(&layer);
        return 1;
    }

    srand(input_seed);
    for (size_t i = 0; i < n_in; i++) {
        x[i] = rand_uniform(-1.0f, 1.0f);
    }

    mamba2_layer_reset_state(&layer);
    mamba2_layer_forward(&layer, x, seq_len, y);

    bool finite = all_finite(y, n_in);
    float mabs = mean_abs(y, n_in);

    printf("mean(|y|)=%.6f\n", mabs);
    printf("all_finite=%s\n", finite ? "true" : "false");

    if (finite && mabs > 0.0f) {
        printf("Test PASSED.\n");
    } else {
        printf("Test FAILED.\n");
        free(x);
        free(y);
        mamba2_layer_free(&layer);
        return 1;
    }

    free(x);
    free(y);
    mamba2_layer_free(&layer);
    return 0;
}
#endif
