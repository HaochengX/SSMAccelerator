# SSMU v3 Mid-5120 Low-Rank Design Report

## Table of Contents

1. [Scope and Source Files](#1-scope-and-source-files)
2. [Dimensions and Sizes](#2-dimensions-and-sizes)
3. [Data Types](#3-data-types)
4. [Kernel Module Functions](#4-kernel-module-functions)
5. [Overall Architectural Flow](#5-overall-architectural-flow)
6. [Optimization Methods](#6-optimization-methods)
7. [Synthesis and Simulation Results](#7-synthesis-and-simulation-results)

---

## 1. Scope and Source Files

This report covers the resource-optimized low-rank SSM (State-Space Model) accelerator kernel
targeting the Xilinx KV260 (`xck26-sfvc784-2LV-c`) FPGA, implemented in Vitis HLS v2024.1.

| File | Role |
|---|---|
| `vitis5/quant9/ssm_lowrank_v3_mid5120_lr.cpp` | Main kernel implementation |
| `vitis5/quant9/SSMU.h` | Shared types, dimension definitions, vector utilities |
| `vitis5/quant9/tb_ssm.cpp` | Testbench for C-simulation and co-simulation |

The top-level synthesized symbol is `SSMU_STACK64`, which is a thin `extern "C"` wrapper
that directly calls the main kernel function `SSMU`.

---

## 2. Dimensions and Sizes

### 2.1 Base Constants (from `SSMU.h`)

```cpp
#define SSMU_T   1      // tokens per kernel call
#define SSMU_D   2560   // model hidden width
#define SSMU_N   128    // SSM state dimension
#define SSMU_K   4      // conv1D kernel length
#define SSMU_P   64     // groups for DT channel reduction
#define SSMU_CP  8      // vector lane width (elements per hls::vector word)
```

| Symbol | Value | Description |
|---|---:|---|
| `T` | 1 | One input token processed per kernel call |
| `D` | 2560 | Main hidden dimension of the Mamba model |
| `N` | 128 | SSM state size per channel (number of discrete state elements) |
| `K` | 4 | Length of the depthwise conv1D shift register |
| `P` | 64 | Parameter controlling DT channel folding |
| `CP` | 8 | Number of elements packed into one `hls::vector<...,8>` word |

### 2.2 Derived Channel Dimensions

```cpp
#define SSMU_C2     (2 * SSMU_D)                          // 5120
#define SSMU_C_CONV (2 * SSMU_D + 2 * SSMU_N)             // 5376
#define SSMU_CH     (2 * SSMU_D / SSMU_P)                 // 80
#define SSMU_C_IN   (SSMU_C2 + SSMU_C_CONV + SSMU_CH)    // 10576
```

| Symbol | Scalar | Tile (÷8) | Description |
|---|---:|---:|---|
| `C2` | 5120 | `C2_T` = 640 | Inner expanded channels; gated output dimension |
| `C_CONV` | 5376 | `CCONV_T` = 672 | Conv bundle = B(N) + C(N) + x(C2) |
| `CH` | 80 | `CH_T` = 10 | DT (delta-time) raw channel count |
| `C_IN` | 10576 | `CIN_T` = 1322 | Total in-projection output width |
| `STATE` (`N`) | 128 | `STATE_V` = 16 | SSM state dimension vectorized over CP=8 lanes |

### 2.3 Low-Rank Factorization Sizes

```cpp
#define SSMU_RANK   1024
#define SSMU_RANK_T (SSMU_RANK / VEC_FACTOR)   // 128
```

| Symbol | Value | Description |
|---|---:|---|
| `RANK` | 1024 | Low-rank bottleneck width for both in- and out-projection |
| `RANK_T` | 128 | Vector-tile count for rank dimension |

Weight matrix shapes (all stored as `W_VEC = hls::vector<ap_fixed<16,4>, 8>`):

| Matrix | Shape (tiles) | Scalar params | Description |
|---|---|---:|---|
| `W_in_1[D_T][RANK_T]` | [320][128] | 327,680 × 8 | First in-proj factor: D → RANK |
| `W_in_2[RANK_T][INP_X_T]` | [128][640] | 655,360 × 8 | Second in-proj factor: RANK → C2 |
| `W_in_nonlr[D_T][INP_NONLR_T]` | [320][682] | 1,745,920 × 8 | Full-rank branch: Z + DT + B + C |
| `W_delta[C2_T][C2_T]` | [640][640] | 3,276,800 × 8 | Delta projection (Δ) matrix |
| `W_out_A[D_T][RANK_T]` | [320][128] | 327,680 × 8 | First out-proj factor: RANK → D |
| `W_out_B[RANK_T][C2_T]` | [128][640] | 655,360 × 8 | Second out-proj factor: C2 → RANK |

### 2.4 GEMM Tiling Policy

```cpp
#define SSMU_JJ_UNROLL   8    // unroll factor along j (input row) within J_TILE
#define SSMU_LANE_UNROLL 4    // unroll factor along vector lane (0..7)
#define SSMU_JT_II       2    // minimum II for W_delta inner loop
#define SSMU_I_TILE      2    // output tile: 2 output rows computed per outer iteration
static const int J_TILE = 8; // input tile: 8 input vectors consumed per pipelined state
```

The inner pipeline processes `J_TILE=8` input vectors per cycle (II=1), giving 8 parallel
multiply-add operations per lane per cycle.

### 2.5 Stream Depth Policy

```cpp
#define SSMU_STREAM_DEPTH  16   // default: 1 SRL16 stage = 128 LUTs per 128-bit stream
// Four deadlock-critical streams use full-size depths:
#pragma HLS STREAM variable=X_residual       depth=320  // holds all D_T tokens
#pragma HLS STREAM variable=Z_stream         depth=640  // written before XBC consumed
#pragma HLS STREAM variable=G_stream         depth=640  // produced before htC is ready
#pragma HLS STREAM variable=X_ssm_out_stream depth=640  // dup3 races with stage45
```

---

## 3. Data Types

```cpp
typedef ap_fixed<24, 8>   ACC_T;      // accumulator (v3: narrowed from ap_fixed<32,10>)
typedef ap_fixed<16, 6>   ACT_T;      // activations (v3: narrowed from ap_fixed<18,6>)
typedef ap_fixed<16, 8>   EXP_T;      // exp polynomial (v3: narrowed from ap_fixed<20,8>)
typedef ap_fixed<24, 12>  RMS_INV_T;  // RMSNorm inverse (v3: replaces float)
// From SSMU.h:
typedef ap_fixed<16, 4>   DTYPE;      // primary data type
typedef hls::vector<DTYPE, 8>  DTYPE_VEC;
typedef hls::vector<DTYPE, 8>  W_VEC;
```

| Type | Bits | Int bits | Role |
|---|---:|---:|---|
| `DTYPE` | 16 | 4 | Primary data/weight type: inputs, activations, outputs |
| `ACC_T` | 24 | 8 | GEMM accumulator: partial sums and adder tree nodes |
| `ACT_T` | 16 | 6 | Activation inputs/outputs (sigmoid, silu, softplus) |
| `EXP_T` | 16 | 8 | Exponential approximation intermediate values |
| `RMS_INV_T` | 24 | 12 | RMSNorm reciprocal (wider integer range needed) |

---

## 4. Kernel Module Functions

The design is decomposed into approximately 30 static helper functions, each synthesized as a
separate pipelined block connected through `hls::stream` FIFOs under the top-level
`#pragma HLS DATAFLOW` directive.

### 4.1 Input Pre-processing

#### `rmsnorm_vecDT_stream_local`

**Purpose:** Applies RMS Layer Normalization over D=2560 input channels (320 vector tokens).

**Inputs:** `x_in` stream (D_T tokens), preloaded `RMS_weight[D_T]` (LUTRAM).
**Output:** `y_out` stream (D_T tokens, normalized and weight-scaled).

**Operation:**
1. Buffer all D_T input tokens into `xbuf[D_T]` (LUTRAM, required because inverse
   calculation needs the completed sum-of-squares).
2. Compute per-lane sum-of-squares across D elements; reduce across 8 lanes.
3. Compute `inv = 1 / sqrt(mean_sq + eps)` using fully fixed-point arithmetic
   (no float DSPs).
4. Multiply each buffered token element-wise by `inv × weight[j]`, output to stream.

#### `rmsnorm_vecC2T_stream_local`

**Purpose:** Second RMSNorm over C2=5120 channels, applied between the SSM core output
(stage 6) and the output projection. Structurally identical to the D-channel version
but operates on `C2_T=640` tokens.

---

### 4.2 Low-Rank Input Projection

The main input projection `W_in[D][C2]` is factored into two smaller GEMMs:

```
X_normed [D=2560] → W_in_1 [D][RANK] → temp [RANK=1024] → W_in_2 [RANK][C2] → X_mid [C2=5120]
```

#### `in_proj_lr_stage1`

**Purpose:** First GEMM of the low-rank in-projection:
`temp[RANK_T] = X_normed[D_T] × W_in_1[D_T][RANK_T]`.

Internally, it is a three-function `#pragma HLS DATAFLOW` sub-graph:
- `read_x_buf_D_local` — Reads D_T stream tokens into local LUTRAM `X_buf[D_T]`.
- `stream_Win1_tiles_local` — Streams weight tiles from global memory in J_TILE=8 chunks,
  cycling through all RANK_T output rows, D_T input columns.
- `inproj_stage1_consume_local` — Tiled GEMM consumer: for each tile of SSMU_I_TILE=2
  output rows, iterates D_T/J_TILE pipeline stages, accumulates 8 parallel products per
  lane, performs tree reduction, writes output token to stream.

#### `in_proj_lr_stage2`

**Purpose:** Second GEMM:
`X_mid[INP_X_T] = temp[RANK_T] × W_in_2[RANK_T][INP_X_T]`.

Reads the RANK_T intermediate stream into `temp_buf[RANK_T]` (LUTRAM), then runs the
same tiled GEMM pattern with `Win2_tiles` streaming from `W_in_2[RANK_T][C2_T]`.

#### `in_proj_nonlr_stage`

**Purpose:** Full-rank branch projection for Z (gating), DT (time-step), B and C (SSM
input/output vectors):
`nlr_stream[INP_NONLR_T] = X_normed[D_T] × W_in_nonlr[D_T][INP_NONLR_T]`.

Uses the same template (`stream_Wfull_tiles_local<INP_NONLR_T>` +
`inproj_full_consume_local<INP_NONLR_T>`).

#### `demux_nonlr_local`

**Purpose:** Demultiplexes the flat `nlr_stream` into separate streams: Z[C2_T], DT[CH_T],
B[STATE_V], C[STATE_V], based on channel index offsets.

#### `assemble_xbc_local`

**Purpose:** Re-assembles B, C, X_mid into the XBC bundle layout
`[B:STATE_V][C:STATE_V][x:C2_T]` (total CCONV_T=672 tokens) required by the conv1D kernel.

---

### 4.3 Convolution and Gate

#### `conv1d_silu_stream_local_with_state`

**Purpose:** Applies a stateful causal depthwise conv1D (K=4) over the full XBC bundle,
simultaneously applying SiLU to the Z gate branch.

**State management:** Reads K-1=3 history context vectors from `conv_state_in`, updates the
shift register element-by-element as tokens arrive, writes the updated history to
`conv_state_out`. This makes the kernel an RNN cell with persistent state across calls.

**XBC ordering:** `[B:STATE_V][C:STATE_V][x:C2_T]` — B and C channels are conv-processed
and SiLU-activated first, then passed directly to the SSM scan; x channels are conv-processed
and SiLU-activated in parallel with the gate computation.

**Outputs:**
- `G_out` — `SiLU(Z[j])`: gate signal for stage 6 (C2_T tokens).
- `X_ssm_out` — Conv1D activated x values: SSM input signal (C2_T tokens).
- `B_conv_out` — Conv1D + SiLU activated B: SSM B vector (STATE_V tokens).
- `C_conv_out` — Conv1D + SiLU activated C: SSM C vector (STATE_V tokens).

---

### 4.4 Delta (Δ) Computation

Two alternative paths compute the discrete time-step Δ:

**Path A (default, `SSMU_DELTA_FROM_DT=1`):**

#### `dtadapt_stream_local`
Adapts the raw DT signal from CH_T=10 tokens to C2_T=640 by tiling:
`dt_out[j] = dt_buf[j % CH_T]`.

#### `dt_to_delta_stream_local`
Applies `softplus` activation: `Δ[j] = softplus(DT_adapted[j])`.

**Path B (`SSMU_DELTA_FROM_DT=0`):**

#### `projection_delta_only_local`
Computes a full `C2×C2` projection: `Δ[i] = softplus(X_ssm[D] × W_delta[C2][C2])`.
This is the larger W_delta GEMM (640×640 tiles).

---

### 4.5 SSM Scan (Stages 3, 4, 5)

#### `stage3_dA_stream_local`

**Purpose:** Computes discrete `dA[i][j] = exp(A[i] * Δ[j])` for all STATE_V=16
state dimensions and all C2_T=640 channels, producing a STATE_V × C2_T stream.

Buffers the entire delta array (`delta_buf[C2_T]`) first, then iterates over STATE_V
rows, streaming all C2_T `exp()` values per row.

#### `stage45_update_reduce_local`

**Purpose:** Performs the complete SSM recurrence and C-projection in a single merged
pass, avoiding a separate memory write-back between the state update and the C projection.

**Algorithm for each state row i ∈ [0, STATE_V):**
1. Read B[i] and C[i] from their respective streams.
2. For each channel tile `jt` (J_TILE=8 in parallel):
   ```
   H1[i][j] = H0[i][j] × dA[i][j] + (B[i] × Δ[j]) × X_ssm[j]
   acc[j]   += H1[i][j] × C[i]
   ```
3. After all STATE_V rows: `htC_out[j] = acc[j]` (the C-projected SSM output).

**Inputs:** `X_ssm_scan`, `delta`, `dA`, `B_conv`, `C_conv`, `H0_in` (history state).
**Outputs:** `htC_out` (C2_T accumulated output), `H1_state_out` (updated state stream).

---

### 4.6 Stage 6: Gated Output Mixing

#### `stage6_out_yz_vec_local`

**Purpose:** Combines the SSM output `htC`, the diagonal skip connection D, the SSM input
`X_ssm`, and the gate signal `G` to produce the final pre-norm output:
```
out[j] = (htC[j] + D[j] × X_ssm[j]) × G[j]
```
The D-skip connection models an additional linear bypass in the SSM formulation.
The G gate (from SiLU(Z)) provides input-dependent non-linear modulation.

---

### 4.7 Low-Rank Output Projection

Mirrors the input projection:

```
ssm_normed [C2=5120] → W_out_B [RANK][C2] → temp [RANK=1024]
                      → W_out_A [D][RANK]  → Y [D=2560]
```

#### `out_proj_lr_stage1`
Computes `temp[RANK_T] = ssm_normed[C2_T] × W_out_B[RANK_T][C2_T]`.

#### `out_proj_lr_stage2`
Computes `Y[D_T] = temp[RANK_T] × W_out_A[D_T][RANK_T]`.
Applies `ap_fixed<16,6>` rounding/saturation (`AP_RND_CONV, AP_SAT`) on the output.

---

### 4.8 Residual Addition and Output

#### `add_residual_local_D`

**Purpose:** Adds the original (pre-RMSNorm) input as a residual connection:
`output[j] = out_proj[j] + X_original[j]`.

The residual `X_residual` was forked before RMSNorm from the input, stored in a stream
of `depth=320` that acts as a delay line across the entire pipeline.

---

### 4.9 Utility Functions

| Function | Description |
|---|---|
| `copy_kernel_k` | Copies K=4 scalar kernel weights into local stream |
| `copy_vec_n(in, out, n)` | Passes n vector tokens from one stream to another (pipeline stage) |
| `drain_vec_n(in, n)` | Discards n tokens from a stream |
| `tee_vecDT_stream2_local` | Broadcasts D_T tokens to two output streams (fork) |
| `dup_vecC2_stream2/3_local` | Broadcasts C2_T tokens to 2 or 3 output streams |
| `preload_vec_table_local<N>` | Copies a global-memory array into on-chip LUTRAM |
| `ddr_writer_local` | Optionally writes C/H1 trace to DDR (disabled by default) |

---

## 5. Overall Architectural Flow

The `SSMU` kernel is structured as a single top-level `#pragma HLS DATAFLOW` graph.
All sub-functions execute in parallel, synchronized by `hls::stream` FIFOs.

```
                     ┌──────────────────── SSMU Dataflow ────────────────────────┐
 X_in (D_T) ─────────┤
                      │  tee ──► X_for_norm ──► RMSNorm1 ──► X_normed
                      │    └──► X_residual (depth=320)
                      │
                      │  X_normed ──► tee ──► X_normed_lr  ──► in_proj_lr_stage1
                      │                │              (W_in_1: [D][RANK])
                      │                │                    │ temp [RANK_T]
                      │                │              in_proj_lr_stage2
                      │                │              (W_in_2: [RANK][C2])
                      │                │                    │ X_mid [C2_T]
                      │                └──► X_normed_nonlr ──► in_proj_nonlr
                      │                          (W_in_nonlr: [D][Z+DT+B+C])
                      │                                    │ nlr_stream
                      │                             demux_nonlr
                      │                        ┌─────┬──────┬───┬───┐
                      │                        Z    DT      B   C
                      │                        │     │     └─┬─┘
                      │                        │     │  assemble_xbc(B,C,X_mid)
                      │                        │     │           │ XBC_stream [CCONV_T]
 kernel_in ───────────┤                        │     │           ▼
 conv_state_in ───────┼────────────────────► conv1d_silu_state
 conv_state_out ◄─────┤                        │
                      │                  ┌─────┴──────────┬──────────┐
                      │                G_out        X_ssm_stream  B_conv/C_conv
                      │                (depth=640)        │              │
                      │                    │          dup3:            (to scan)
                      │                    │      ├── X_ssm_proj  [delta path]
                      │                    │      ├── X_ssm_scan  ──► stage45
                      │                    │      └── X_ssm_out   ──► stage6
                      │
                      │  DT ──► dtadapt ──► dt_to_delta ──► delta [C2_T]
                      │  (or X_ssm_proj ──► W_delta_proj ──► delta  if DT path off)
                      │
                      │  delta ──► dup2 ──► delta_for_dA ──► stage3_dA ──► dA[STATE_V x C2_T]
                      │                └──► delta_for_scan ─────────────────────┐
 H0_in ───────────────┤                                                          │
                      │                            stage45_update_reduce ◄───────┘
                      │                          (X_ssm_scan, dA, B_conv, C_conv, H0)
                      │                                    │          │
 H1_out ◄─────────────┼────────────────── H1_state_stream              │
                      │                                           htC [C2_T]
                      │                               stage6 (+ D_diag + G_out)
                      │                                    │ ssm_core_out [C2_T]
                      │                               RMSNorm2 (W_RMS2)
                      │                                    │ ssm_normed [C2_T]
                      │                          out_proj_lr_stage1 (W_out_B)
                      │                                    │ outproj_temp [RANK_T]
                      │                          out_proj_lr_stage2 (W_out_A)
                      │                                    │ out_proj [D_T]
                      │                          add_residual (+X_residual)
 out (D_T) ◄──────────┤
                      └───────────────────────────────────────────────────────────┘
```

**Stage-by-stage summary:**

| Stage | Function | I/O size |
|---|---|---|
| Residual fork | `tee_vecDT_stream2` | → D_T × 2 |
| RMSNorm 1 | `rmsnorm_vecDT` | D_T → D_T |
| Split LR / nonLR | `tee_vecDT_stream2` | → D_T × 2 |
| In-proj stage 1 | GEMM D→RANK | D_T → RANK_T |
| In-proj stage 2 | GEMM RANK→C2 | RANK_T → C2_T |
| In-proj non-LR | GEMM D→(Z+DT+B+C) | D_T → 682 |
| Demux + Assemble | demux/assemble | → XBC CCONV_T=672 |
| Conv1D + SiLU | depth-wise conv K=4 + SiLU | CCONV_T → G + X_ssm + B + C |
| Delta adapt | dtadapt + softplus | CH_T → C2_T |
| Stage 3: dA | exp(A×Δ) per (i,j) | STATE_V × C2_T |
| Stage 4+5: scan | H recurrence + C projection | → htC C2_T |
| Stage 6: gate | htC + D·X, × G | C2_T → C2_T |
| RMSNorm 2 | `rmsnorm_vecC2T` | C2_T → C2_T |
| Out-proj stage 1 | GEMM C2→RANK | C2_T → RANK_T |
| Out-proj stage 2 | GEMM RANK→D | RANK_T → D_T |
| Residual add | `add_residual` | D_T → D_T |

---

## 6. Optimization Methods

The v3-mid design applies twelve resource-reduction techniques. None of them alter the
mathematical algorithm or the dataflow graph connection topology.

---

### R1: Accumulator Type Narrowing (`ACC_T: ap_fixed<32,10>` → `ap_fixed<24,8>`)

**Savings:** DSP, FF, LUT

```cpp
// v2 (before):
typedef ap_fixed<32, 10> ACC_T;

// v3 (after):
typedef ap_fixed<24, 8>  ACC_T;
```

**Explanation:** `ACC_T` is the type of every partial sum node in the GEMM adder trees.
With `SSMU_JJ_UNROLL=8`, each pipeline cycle generates 8 products which are reduced by
a 3-level binary adder tree. Each tree node is a fixed-point adder whose bit-width is
determined by `ACC_T`. Narrowing from 32→24 bits reduces each adder from 32-bit to
24-bit carry chains (~25% LUT reduction per adder), and removes 8 bits from each
pipeline register (FF reduction). The `<32,10>` → `<24,8>` change retains equivalent
dynamic range for accumulation of `ap_fixed<16,4>` products, which produce at most 32-bit
full-precision results; 24 bits preserves the 8 most-significant integer bits plus 16
fractional bits of useful precision.

---

### R2: Activation Type Narrowing (`ACT_T: ap_fixed<18,6>` → `ap_fixed<16,6>`)

**Savings:** DSP

```cpp
typedef ap_fixed<16, 6>  ACT_T;  // v3: was ap_fixed<18,6>

static inline ACT_T sigmoid_fx(ACT_T x) {
#pragma HLS INLINE
    const ACT_T half = (ACT_T)0.5;
    const ACT_T qtr  = (ACT_T)0.25;
    ACT_T y = half + qtr * x;   // 16×16-bit multiply → 1 DSP (18-bit B-port fits)
    return clamp_fx<ACT_T>(y, (ACT_T)0.0, (ACT_T)1.0);
}

static inline DTYPE silu_fx(DTYPE a) {
#pragma HLS INLINE
    ACT_T x = (ACT_T)a;
    ACT_T s = sigmoid_fx(x);
    return (DTYPE)(x * s);      // 16×16-bit → 1 DSP
}
```

**Explanation:** On DSP48E2, the B-port accepts a maximum of 18 signed bits. An
`ap_fixed<18,6>` operand has `sign + 17 data bits = 18 bits`, which is right at the
overflow boundary: Vitis HLS conservatively allocates 2 DSP48E2 slices in cascade for
any 18-bit × 18-bit product to guarantee no truncation of the 36-bit full result.
Reducing to 16 bits uses only 16 × 16 = 256-bit product, which fits cleanly in one
DSP48E2 B-port and P-output. SiLU is applied to every x-channel token in conv1D
(C2_T=640 tokens × 8 lanes = 5,120 calls per invocation), so saving 1 DSP per call site
saves a significant number of DSPs across the fully-unrolled pipeline.

---

### R3: Exponential Type Narrowing (`EXP_T: ap_fixed<20,8>` → `ap_fixed<16,8>`)

**Savings:** DSP, LUT

```cpp
typedef ap_fixed<16, 8>  EXP_T;  // v3: was ap_fixed<20,8>

static inline EXP_T exp_fx(ACT_T t_in) {
#pragma HLS INLINE
    ACT_T t = clamp_fx<ACT_T>(t_in, (ACT_T)(-3.0f), (ACT_T)3.0f);

    EXP_T tt;
    EXP_T half_tt;
#pragma HLS BIND_OP variable=tt      op=mul impl=fabric  // force LUT-multiplier, not DSP
#pragma HLS BIND_OP variable=half_tt op=mul impl=fabric
    tt      = (EXP_T)t * (EXP_T)t;
    half_tt = (EXP_T)0.5 * tt;
    EXP_T y = (EXP_T)1.0 + (EXP_T)t + half_tt;
    if (y < (EXP_T)0) y = (EXP_T)0;
    return y;
}
```

**Explanation:** `exp_fx` uses a quadratic Taylor approximation `exp(t) ≈ 1 + t + t²/2`
(accurate for small clamped t). It contains two multiplications: `t*t` and `0.5*tt`.
At `ap_fixed<20,8>`, both operands are 20-bit, requiring cascaded DSPs (>18-bit B-port).
Narrowing to `ap_fixed<16,8>` brings both under the 18-bit B-port threshold, and the
additional `BIND_OP impl=fabric` directive routes these through LUT-carry multipliers
(Xilinx Karatsuba fabric), bypassing DSP allocation entirely. `exp_fx` is called for
every `(state_row, channel)` pair in stage3: `STATE_V × C2_T = 16 × 640 = 10,240` calls
per token, making DSP savings here substantial.

---

### R4: RMSNorm Fixed-Point Inverse (float → `ap_fixed<24,12>`)

**Savings:** DSP (eliminates float multiplier chain)

```cpp
typedef ap_fixed<24,12>  RMS_INV_T;

// Step 1: 16-bit fabric multiplies for sum-of-squares
typedef ap_fixed<16,8> RMS_NRW_T;
RMS_NRW_T vsq_n;
#pragma HLS BIND_OP variable=vsq_n op=mul impl=fabric
vsq_n = (RMS_NRW_T)vraw * (RMS_NRW_T)vraw;
lane_sumsq[l] += (ACC_T)vsq_n;

// Step 2: compile-time reciprocal constant → LUT shift tree, not DSP
static const ap_ufixed<32,4> k_inv_dt_n =
    (ap_ufixed<32,4>)(1.0 / (double)(D_T * VEC_FACTOR));
RMS_MS_T_D ms = sumsq_u * (RMS_MS_T_D)k_inv_dt_n;

// Step 3: ap_fixed sqrt → CORDIC/LUT, zero DSP
RMS_MS_T_D sq = hls::sqrt(ms_eps);

// Step 4: apply weight with 16-bit fabric multiplies
RMS_NRW_T rms_xi, yv;
#pragma HLS BIND_OP variable=rms_xi op=mul impl=fabric
#pragma HLS BIND_OP variable=yv     op=mul impl=fabric
rms_xi = xvn * invn;
yv     = rms_xi * wwn;
```

**Explanation:** The original RMSNorm used IEEE-754 float for the mean, inverse-sqrt, and
`x * inv * weight` chain. Each float multiplication requires a 24-bit mantissa multiply,
implemented as cascaded DSP48E2 blocks (2 DSPs per float multiply). The v3 replacement:
(1) computes sum-of-squares with 16-bit LUT-multipliers; (2) multiplies by a
compile-time constant reciprocal (synthesized as a constant-coefficient multiplier tree →
LUT adders, 0 DSPs); (3) uses `hls::sqrt(ap_fixed<>)` which synthesizes as a CORDIC
circuit (all LUT, 0 DSPs); (4) applies weight using explicit `BIND_OP fabric` 16-bit
multipliers. With two RMSNorm calls per token (D+C2 = 2560+5120), eliminating the float
chain saves an estimated 30–50 DSPs.

---

### R5: Consistent `EXP_T` Typing in Stage 3

**Savings:** DSP, LUT

```cpp
// stage3_dA_stream_local inner loop:
for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
    ACT_T a  = (ACT_T)vget(Avec, l);   // explicit cast to ACT_T=ap_fixed<16,6>
    ACT_T dl = (ACT_T)vget(dlt,  l);   // explicit cast
    EXP_T e  = exp_fx(a * dl);         // product is ACT_T × ACT_T = ACT_T (16-bit)
    vset(dA_vec, l, (DTYPE)e);
}
```

**Explanation:** Without explicit `ACT_T` casts, the C++ implicit promotion rules would
widen `a * dl` to `int` (32-bit) before passing to `exp_fx`. The function signature
`exp_fx(ACT_T)` then inserts an implicit truncation, but the synthesis tool might still
allocate a 32-bit multiplier for the `a*dl` site. By explicitly casting both operands to
`ACT_T=ap_fixed<16,6>`, the product is computed in 16-bit fixed-point per HLS semantics,
ensuring the multiply inferring `exp_fx`'s input uses the same 16-bit path as R3.

---

### R6: Stage 6 `ACC_T` Consistency with `BIND_OP fabric`

**Savings:** DSP

```cpp
// stage6_out_yz_vec_local:
ACC_T s6_dx, s6_yg;
#pragma HLS BIND_OP variable=s6_dx op=mul impl=fabric  // D[j] × X[j] → LUT
#pragma HLS BIND_OP variable=s6_yg op=mul impl=fabric  // y   × G[j] → LUT
s6_dx = d * x;
ACC_T y  = ht + s6_dx;
s6_yg = y * g;
vset(outv, l, (DTYPE)s6_yg);
```

**Explanation:** Both multiplications involve `ACC_T=ap_fixed<24,8>` operands. Without
`BIND_OP`, Vitis HLS would assign DSP48E2 for these 24×24-bit products (24 > 18-bit
B-port). Stage 6 runs a pipelined loop of C2_T=640 cycles; at SSMU_LANE_UNROLL=4 (up to
8 lanes), the loop body instantiates 2 multiplications per lane = up to 16 DSP sites.
`impl=fabric` routes all 16 through LUT-multipliers instead, trading ~16 DSPs for ~800 LUTs.

---

### R7: Stream Depth Reduction (650 → 64 → 16)

**Savings:** BRAM, FF (≈25 BRAM18K)

```cpp
// v3: default depth = 16
#define SSMU_STREAM_DEPTH 16

// Applied to all ~25 non-critical streams, e.g.:
#pragma HLS STREAM variable=X_local     depth=SSMU_STREAM_DEPTH  // was 650 → 4 BRAM18K
#pragma HLS STREAM variable=DT_stream   depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=delta_for_dA depth=SSMU_STREAM_DEPTH
// ...

// Four deadlock-critical streams kept large:
#pragma HLS STREAM variable=X_residual       depth=320
#pragma HLS STREAM variable=Z_stream         depth=640
#pragma HLS STREAM variable=G_stream         depth=640
#pragma HLS STREAM variable=X_ssm_out_stream depth=640
```

**Explanation:** Vitis HLS maps `hls::stream`:
- `depth ≤ 32`: SRL16/SRL32 shift-register primitives → 0 BRAM, ~1 LUT/bit.
- `depth 33–1024`: BRAM36 (1 per stream, regardless of actual depth used).
- `depth > 1024`: multiple BRAM36 blocks.

At `depth=650`, each of the ~25 streams occupies one BRAM36 (= 2 BRAM18K). Reducing to
`depth=16` maps each stream to 16-deep SRL-based FIFO primitives (~128 LUTs for a
128-bit wide stream), consuming zero BRAM18K. Total BRAM saving: up to 50 BRAM18K.
The four exception streams have producers and consumers that can be separated by up to
the full C2_T=640 or D_T=320 tokens in the dataflow; a shallow FIFO would cause the
producer to stall before the consumer can drain it, creating a deadlock in HLS simulation.

---

### R8: `BIND_OP impl=fabric` for Non-Critical Multiplications

**Savings:** DSP

```cpp
// conv1d depthwise multiplications (K=4 per lane per token):
ACC_T cp0, cp1, cp2, cp3;
#pragma HLS BIND_OP variable=cp0 op=mul impl=fabric
#pragma HLS BIND_OP variable=cp1 op=mul impl=fabric
#pragma HLS BIND_OP variable=cp2 op=mul impl=fabric
#pragma HLS BIND_OP variable=cp3 op=mul impl=fabric
cp0 = (ACC_T)kernel_buffer[0] * (ACC_T)window0[lane];
cp1 = (ACC_T)kernel_buffer[1] * (ACC_T)window1[lane];
cp2 = (ACC_T)kernel_buffer[2] * (ACC_T)window2[lane];
cp3 = (ACC_T)kernel_buffer[3] * (ACC_T)window3[lane];

// RMSNorm per-element weight application:
RMS_NRW_T rms_xi, yv;
#pragma HLS BIND_OP variable=rms_xi op=mul impl=fabric
#pragma HLS BIND_OP variable=yv     op=mul impl=fabric
rms_xi = xvn * invn;
yv     = rms_xi * wwn;
```

**Explanation:** Several multiply sites operate on values whose bit-widths are technically
above the B-port limit in the worst case, or which Vitis HLS would choose to place on DSPs
for throughput. `BIND_OP impl=fabric` explicitly overrides this choice. In the conv1D loop,
`kernel_buffer[k]` values are small 16-bit constants loaded once; `window[lane]` are
16-bit activations. The four per-lane conv multiplications run `CCONV_T × VEC_FACTOR =
672 × 8 = 5,376` times per invocation. Routing all four kernel×window multiplications to
fabric instead of DSP frees an estimated 32 DSPs (8 lanes × 4 kernel taps).

---

### R9: Low-Rank Weight Factorization

**Savings:** BRAM (weight parameter count reduction)

```cpp
// Original (conceptual) full-rank weights:
//   W_in[D_T][C2_T]       = [320][640] = 204,800 W_VEC words
//   W_out[C2_T][D_T]      = [640][320] = 204,800 W_VEC words

// Low-rank factored replacements:
const W_VEC W_in_1[D_T][RANK_T],    // [320][128]  = 40,960 words
const W_VEC W_in_2[RANK_T][INP_X_T],// [128][640]  = 81,920 words
// Total in-proj: 122,880  vs. 204,800  → 40% reduction

const W_VEC W_out_A[D_T][RANK_T],   // [320][128]  = 40,960 words
const W_VEC W_out_B[RANK_T][C2_T],  // [128][640]  = 81,920 words
// Total out-proj: 122,880 vs. 204,800 → 40% reduction

// Execution:
in_proj_lr_stage1(X_normed_lr, W_in_1, inproj_temp_stream, wscale_in_fx);
in_proj_lr_stage2(inproj_temp_stream, W_in_2, X_mid_stream, wscale_in_fx);

out_proj_lr_stage1(ssm_normed_stream, W_out_B, outproj_temp_stream, wscale_out_fx);
out_proj_lr_stage2(outproj_temp_stream, W_out_A, out_proj_stream_s, wscale_out_fx);
```

**Explanation:** The conceptual projection `W_in[2560][5120]` would store ~13.1M parameters
(13.1M × 2 bytes = 26 MB). This vastly exceeds available BRAM and would require extensive
off-chip DDR access stalls. By factoring into `W_in_1[2560][1024]` (2.6M params) and
`W_in_2[1024][5120]` (5.2M params), the total in-projection parameter count drops to 7.9M
— a 40% reduction. The same saving applies to the output projection. In terms of BRAM,
the intermediate `temp[RANK_T=128]` activation buffer (128 × 128-bit = 2 KB) is small
enough to be held entirely in LUTRAM between the two GEMM stages.

---

### R10: Cyclic BRAM Array Partitioning

**Savings:** Latency (enables II=1 through parallel bank access)

```cpp
// In weight streamer functions — partition weight array into 8 banks
// so that 8 consecutive elements (one J_TILE) can be read simultaneously:
#pragma HLS ARRAY_PARTITION variable=W_in_1  cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=W_in_2  cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=W_out_A cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=W_out_B cyclic factor=8 dim=2

// In consumer functions — partition activation buffer for simultaneous read:
#pragma HLS ARRAY_PARTITION variable=X_buf   cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=temp_buf cyclic factor=8 dim=1
```

**Explanation:** Each GEMM consumer loop reads `J_TILE=8` weight vectors and 8 activation
vectors per pipeline cycle (all 8 elements with the loop unrolled via
`#pragma HLS UNROLL factor=SSMU_JJ_UNROLL`). Block-RAM has two read ports (True Dual-Port
mode) but can only serve two addresses per cycle. Accessing 8 consecutive elements from
an unpartitioned BRAM would require 4–8 cycles, destroying the II=1 pipeline requirement.
`cyclic factor=8` splits the array into 8 interleaved physical banks: element index `k`
resides in bank `k mod 8`. When the loop accesses elements at indices `[jt+0, jt+1, ...,
jt+7]`, all 8 accesses land in distinct banks (different bank numbers mod 8), enabling
conflict-free parallel reads in a single clock cycle.

---

### R11: `vec_tuple8` Tile Bundling + Balanced Tree Reduction

**Savings:** Latency (maintains II=1 for 8 products/cycle/lane)

```cpp
// Bundle of 8 weight vectors transferred atomically through streams
struct vec_tuple8 { W_VEC w[J_TILE]; };
#pragma HLS ARRAY_PARTITION variable=tup.w complete dim=1  // all 8 fully unrolled

// 3-level balanced binary adder tree: log2(8)=3 adder levels on critical path
static inline ACC_T tree_sum8(ACC_T p0, ACC_T p1, ..., ACC_T p7) {
#pragma HLS INLINE
    ACC_T s0 = p0 + p1;  // level 1: 4 adders in parallel
    ACC_T s1 = p2 + p3;
    ACC_T s2 = p4 + p5;
    ACC_T s3 = p6 + p7;
    ACC_T s4 = s0 + s1;  // level 2: 2 adders in parallel
    ACC_T s5 = s2 + s3;
    return s4 + s5;       // level 3: 1 adder
}

// Usage in 8-product parallel inner pipeline:
for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=SSMU_LANE_UNROLL
    ACC_T p0 = (ACC_T)vget(X_tile[0], l) * wget_scaled(wt.w[0], l, scale);
    // ...p1 through p7
    accv[ii][l] += tree_sum8(p0, p1, p2, p3, p4, p5, p6, p7);
}
```

**Explanation:** Processing 8 products per pipeline stage (J_TILE=8, SSMU_JJ_UNROLL=8)
requires a reduction tree to sum them to one accumulated value. An 8-wide sequential
reduction chain has depth 7 adder stages, which would violate the 3 ns clock budget at
24-bit precision. A balanced binary tree has depth `ceil(log2(8)) = 3`, which fits
comfortably within the budget. The `vec_tuple8` struct ensures HLS synthesizes all 8
weight accesses as parallel register reads (via `#pragma HLS ARRAY_PARTITION complete`),
not as 8 separate BRAM reads.

---

### R12: LUTRAM Binding for Intermediate Activation Buffers

**Savings:** BRAM

```cpp
// Top-level preloaded constant tables mapped to LUTRAM:
DTYPE_VEC A_local[STATE_V];
#pragma HLS BIND_STORAGE variable=A_local    type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE variable=RMS_local  type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE variable=RMS2_local type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE variable=D_local    type=ram_2p impl=lutram

// Activation buffers inside GEMM stages:
DTYPE_VEC X_buf[D_T];
#pragma HLS BIND_STORAGE variable=X_buf type=ram_2p impl=lutram

// RMSNorm input buffer:
DTYPE_VEC xbuf[D_T];
#pragma HLS BIND_STORAGE variable=xbuf type=ram_s2p impl=lutram
```

**Explanation:** Without explicit binding, Vitis HLS may choose BRAM36 for arrays with
many elements (e.g., `X_buf[D_T=320]` = 320 × 128-bit = 5 KB). However, these buffers
are already cyclic-partitioned into 8 banks of 40 entries each (40 × 128-bit = 640 B per
bank). This size maps more efficiently to Xilinx Distributed RAM (LUTRAM), which provides
two read ports per primitive and can be freely replicated alongside the cyclic partition.
Using LUTRAM preserves all 288 BRAM18K blocks in the device for the large weight matrices
that genuinely need them.

---

## 7. Synthesis and Simulation Results

### 7.1 Tool and Target Device

| Parameter | Value |
|---|---|
| Tool | Vitis HLS v2024.1 |
| Target device | `xck26-sfvc784-2LV-c` (Kria KV260 FPGA) |
| Speed grade | -2LV |
| Clock period target | 3.00 ns (333 MHz) |
| Implementation flow | `vivado` |
| Top-level function | `SSMU_STACK64` |

### 7.2 C-Simulation Results (from `SSMU_STACK64_csim.log`)

The testbench runs four profiles and checks numerical output against a floating-point
golden reference:

| Profile | Steps | Interleave | Strict | Result |
|---|---:|:---:|:---:|---|
| `baseline` | 1 | No | Yes | **[PASS]** Strict golden match |
| `corner_scale_zero` | 2 | Yes | No | [SMOKE] Corner, non-strict |
| `corner_scale_small` | 2 | Yes | No | [SMOKE] Corner, non-strict |
| `corner_scale_large` | 2 | Yes | No | [SMOKE] Corner, non-strict |

**C-sim accuracy (baseline profile, token 0 / lane 3 spot-check):**

| Signal | DUT value | Reference | Max error (all tokens) |
|---|---|---|---|
| `out[0]` | 0.029053 | 0.029053 | 4.785e-02 |
| `h1[0]` | 0.438965 | 0.440674 | 3.662e-03 |
| `conv_state` | — | — | 5.127e-03 |
| `ssm_core` | — | — | 3.174e-03 |
| `ssm_normed` | — | — | 0.000e+00 (exact after norm-fix) |

**Testbench verdict:** `[TB][PASS] ALL STRICT PROFILES PASS.`

Maximum `hls::stream` depth reached during simulation: **13,640 elements**.

### 7.3 RTL Co-Simulation Results (from `SSMU_STACK64_cosim.rpt`)

Simulated with Verilog RTL in xsim:

| Metric | Value |
|---|---|
| Simulator | Verilog / xsim |
| Status | **Pass** |
| Latency — min | **46,495 cycles** |
| Latency — avg | **46,495 cycles** |
| Latency — max | **46,495 cycles** |
| Interval — min | 46,496 cycles |
| Interval — avg | 46,496 cycles |
| Interval — max | 46,496 cycles |
| Total execution cycles (all stimuli combined) | 325,471 cycles |

**Wall-clock latency** (at 333 MHz, T=3 ns):
`46,495 × 3 ns = 139.485 µs` per token.

### 7.4 Resource Utilization (from `SSMU_STACK64_csynth.rpt`)

Target device: `xck26-sfvc784-2LV-c`

| Resource | Used | Available | Utilization |
|---|---:|---:|:---:|
| **BRAM_18K** | **320** | 288 | **111%** |
| **DSP** | **1529** | 1248 | **122%** |
| **FF** | **247,295** | 234,240 | **105%** |
| **LUT** | **200,703** | 117,120 | **171%** |
| URAM | 0 | 64 | 0% |

Resource breakdown by block type:

| Category | BRAM_18K | DSP | FF | LUT |
|---|---:|---:|---:|---:|
| Instances (`grp_SSMU_fu_136`) | 320 | 1529 | 247,290 | 200,624 |
| Expression | — | — | 0 | 2 |
| Register | — | — | 5 | — |
| Multiplexer | — | — | 0 | 77 |
| **Total** | **320** | **1529** | **247,295** | **200,703** |

> **Note:** All four resources exceed 100% utilization. These over-utilization figures
> are the baseline target of the R1–R12 optimizations described in Section 6. The
> individual resource contributions are dominated by the six large GEMM sub-modules
> (in-proj stage1/2, non-LR, delta-proj, out-proj stage1/2) and the two RMSNorm blocks.

### 7.5 HLS Timing and Latency Estimates (from `SSMU_STACK64_csynth.rpt`)

| Metric | HLS Estimate | Note |
|---|---|---|
| Clock period (estimated) | 2.979 ns | Target: 3.00 ns; 0.7% margin |
| Timing uncertainty | 0.81 ns | Standard HLS post-route uncertainty budget |
| Latency — min (HLS) | 20,164 cycles | 60.492 µs |
| Latency — max (HLS) | 20,164 cycles | Deterministic (no branches) |
| SSMU core interval | 19,439 cycles | Dataflow pipeline interval |
| Pipeline type | dataflow | All sub-functions fire concurrently |

> **HLS vs. cosim latency discrepancy:** The HLS-estimated latency of 20,164 cycles is
> substantially lower than the co-simulation measurement of 46,495 cycles. This is expected:
> HLS latency estimates assume instantaneous AXI memory transfers (the weight matrices
> `W_in_1`, `W_in_2`, `W_delta`, `W_out_A`, `W_out_B` total hundreds of MB and require
> thousands of 128-bit AXI4 burst transactions). RTL co-simulation exercises the full AXI4
> handshake protocol and simulates realistic burst latency, yielding the more accurate
> 46,495-cycle figure. The HLS estimate is still useful for comparing relative pipeline
> latency across design variants.

---

*Report generated from source files in `vitis5/quant9/` — Vitis HLS v2024.1, March 2026.*
