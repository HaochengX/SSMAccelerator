# SSMU v3 Mid-5120 Low-Rank Architecture Guide

Target kernel: `vitis5/quant9/ssm_lowrank_v3_mid5120_lr.cpp`  
Top entry: `SSMU_STACK64` (wrapper of `SSMU`)

---

## 1) Scope and design intent

This kernel is a **v3 resource-optimized pipeline** with a **mid-5120 partial low-rank input projection**:

- Total input-projection output channels = `10576`
- Split strategy = `5120 + 5120 + (80 + 128 + 128)`
- Only the middle `5120` block (`X_mid`) uses low-rank factorization
- `Z` and `DT/B/C` remain full-rank for representation stability
- Output projection stays two-stage low-rank

This gives a practical middle ground between:

- fully full-rank (higher cost), and
- fully low-rank (higher approximation risk).

---

## 2) Dimension reference

### 2.1 Scalar dimensions

| Symbol | Value | Meaning |
|---|---:|---|
| `D` | 2560 | model hidden width |
| `C2` | 5120 | SSM channel width |
| `N` | 128 | state width |
| `CH` | 80 | dt channel width |
| `CP` | 8 | vector lane factor |

### 2.2 Tokenized (vectorized) dimensions

| Symbol | Value | Formula |
|---|---:|---|
| `D_T` | 320 | `D / CP` |
| `C2_T` | 640 | `C2 / CP` |
| `STATE_V` | 16 | `N / CP` |
| `CH_T` | 10 | `CH / CP` |
| `CIN_T` | 1322 | `10576 / CP` |

### 2.3 Input split (token space)

| Segment | Tokens | Scalars |
|---|---:|---:|
| `Z` | `INP_Z_T = 640` | 5120 |
| `X_mid` | `INP_X_T = 640` | 5120 |
| `DT` | `INP_DT_T = 10` | 80 |
| `B` | `INP_B_T = 16` | 128 |
| `C` | `INP_C_T = 16` | 128 |
| non-LR total | `INP_NONLR_T = 682` | 5456 |

---

## 3) Low-rank forms (exact mapping)

### 3.1 Input projection

Low-rank branch (`X_mid` only):

1. `X_normed(2560) x W_in_1(2560x1024) -> temp(1024)`
2. `temp(1024) x W_in_2(1024x5120) -> X_mid(5120)`

So yes, the input low-rank core is exactly **`2560x1024` then `1024x5120`**.

Non-LR branch:

- `X_normed(2560) x W_in_nonlr(2560x5456)`
- demux to `Z(5120), DT(80), B(128), C(128)`

### 3.2 Output projection

Output projection is always two-stage low-rank (not 3-way split):

1. `ssm_normed(5120) x W_out_B(5120x1024) -> temp(1024)`
2. `temp(1024) x W_out_A(1024x2560) -> out_proj(2560)`
3. `out = out_proj + residual`

---

## 4) End-to-end architecture in modules (A to K)

Each module below includes: role, I/O, internal compute, optimization rationale, and handoff.

## Module A — Input copy and residual split

- **Functions**: `copy_vec_n`, `tee_vecDT_stream2_local`
- **Input**: `X_in` (`D_T=320` tokens)
- **Outputs**:
	- `X_for_norm` (main path)
	- `X_residual` (shortcut path)
- **Internal behavior**: token-preserving stream fork
- **Why this design**: residual must keep original scale and ordering
- **Handoff**:
	- `X_for_norm -> Module B`
	- `X_residual -> Module K`

## Module B — RMSNorm1 on `D_T`

- **Function**: `rmsnorm_vecDT_stream_local`
- **Inputs**: `X_for_norm`, `RMS_weight[D_T]`
- **Output**: `X_normed`
- **Compute**:
	- accumulate mean-square over full `D_T`
	- compute inverse RMS
	- apply per-token weight
- **Optimization notes**:
	- fixed-point RMS path (v3 R4)
	- narrowed types for lower arithmetic cost
- **Handoff**: `X_normed -> Module C`

## Module C — Dual-path input projection

### C1 LR path (`X_mid` generation)

- **Functions**: `in_proj_lr_stage1`, `in_proj_lr_stage2`
- **I/O**:
	- stage1: `320 -> 128`
	- stage2: `128 -> 640`
- **Compute pattern**:
	- tiled weight streaming
	- tree reduction on 8-term partial products
- **Key idea**: reduce matrix cost while preserving X-mid capacity

### C2 Non-LR path (`Z/DT/B/C` generation)

- **Functions**: `in_proj_nonlr_stage`, `demux_nonlr_local`
- **I/O**: `320 -> 682` then split to `640/10/16/16`
- **Why non-LR here**: keeps `Z` and control/state factors less approximated

### C3 XBC assembly

- **Function**: `assemble_xbc_local`
- **Order**: `[B][C][X_mid]`
- **Reason**: consistent with downstream conv/demux expectation
- **Handoff**:
	- `XBC -> Module D`
	- `Z, DT -> Modules D/E`

## Module D — Conv1d + gate + conv-state update

- **Function**: `conv1d_silu_stream_local_with_state`
- **Inputs**: `XBC`, `Z`, `kernel`, `conv_state_in`
- **Outputs**: `G`, `X_ssm`, `B_conv`, `C_conv`, `conv_state_out`
- **Compute highlights**:
	- line-buffer based `K=4` 1D conv
	- gate derived from `Z` branch
	- shared stream schedule for B/C/X partitions
- **Optimization notes**:
	- selective fabric multiply binding in conv MAC points
- **Handoff**:
	- `X_ssm -> Modules E/G/H`
	- `B_conv/C_conv -> Module G`
	- `G -> Module H`

## Module E — DT adaptation and delta formation

- **Functions**: `dtadapt_stream_local`, `dt_to_delta_stream_local`
- **Input**: `DT_stream` (`CH_T` scale)
- **Output**: `delta_selected` (`C2_T` aligned)
- **Compute**:
	- remap/broadcast dt channels to `C2_T`
	- softplus transform for positive delta domain
- **Handoff**: `delta_selected -> Modules F/G`

## Module F — Stage3 dA generation

- **Function**: `stage3_dA_stream_local`
- **Inputs**: `delta`, `A_fixed`
- **Output**: `dA_stream`
- **Compute**: per-token exponential transform `exp(A * delta)`
- **Optimization notes**:
	- narrowed `EXP_T`
	- controlled precision in exponential path
- **Handoff**: `dA_stream -> Module G`

## Module G — Stage45 state update and reduction

- **Function**: `stage45_update_reduce_local`
- **Inputs**: `X_ssm`, `delta`, `dA`, `B_conv`, `C_conv`, `H0`
- **Outputs**: `htC`, `H1_state`
- **Compute**:
	- state update across `STATE_V`
	- reduction along state dimension into `htC`
- **Why it matters**: this is the central recurrence/aggregation stage
- **Handoff**:
	- `htC -> Module H`
	- `H1_state -> output stream`

## Module H — Stage6 core output (YZ)

- **Function**: `stage6_out_yz_vec_local`
- **Inputs**: `htC`, `D_diag`, `X_ssm_out`, `G`
- **Formula**: `(htC + D*x) * gate`
- **Output**: `ssm_core_out`
- **Optimization notes**:
	- controlled accumulator precision (v3 R6)
	- avoid unnecessary type widening on critical multiplies
- **Handoff**: `ssm_core_out -> Module I`

## Module I — RMSNorm2 on `C2_T`

- **Function**: `rmsnorm_vecC2T_stream_local`
- **Inputs**: `ssm_core_out`, `RMS_weight_2`
- **Output**: `ssm_normed`
- **Compute role**: final normalization before output projection
- **Debug support**: stage dumps for direct reference comparison
- **Handoff**: `ssm_normed -> Module J`

## Module J — Two-stage LR output projection

### J1 Stage-1 (`W_out_B`): `640 -> 128`

- **Functions**: `read_x_buf_C2_local`, `stream_WoutB_tiles_local`, `outproj_stage1_consume_local`
- **Compute pattern**:
	- preload + tiled weight stream
	- tree-sum8 accumulation
- **Optimization notes**:
	- `J_TILE=8` structured reduction
	- unroll policy for throughput/resource balance

### J2 Stage-2 (`W_out_A`): `128 -> 320`

- **Functions**: `read_temp_buf_RANK_local`, `stream_WoutA_tiles_local`, `outproj_stage2_consume_local`
- **Compute pattern**: same reduction shape as J1 (timing-friendly symmetry)
- **Output quantization**: `ap_fixed<16,6,AP_RND_CONV,AP_SAT>`
- **Debug support**: `dut_out_proj_f32.bin`

## Module K — Residual merge and final output

- **Function**: `add_residual_local_D`
- **Inputs**: projected output + `X_residual`
- **Output**: final `out`
- **Role**: completes residual topology and final stream emission

---

## 5) Chaining and scheduling behavior

Main functional chain:

`A -> B -> C -> D -> (E + F) -> G -> H -> I -> J -> K`

Scheduling-critical coupling points:

- `Z_stream`: large early production, later downstream consumption
- `G_stream`: gate produced before `htC` is ready in stage6
- `X_ssm_out_stream`: 3-way duplication can cause back-pressure if shallow
- `X_residual`: written early, consumed at final merge

These depth assignments are not cosmetic; they are required for robust DATAFLOW execution.

---

## 6) Optimization map (what is optimized and where)

### 6.1 Numeric/path-level optimization

- Narrowed arithmetic types (`ACC_T`, `ACT_T`, `EXP_T`) for lower cost
- Fixed-point RMS normalization paths
- Explicitly controlled precision in stage3 and stage6

### 6.2 Structural optimization

- Partial low-rank at input (`X_mid` only)
- Full low-rank at output projection (two-stage)
- Tile + tree reduction in all major projection kernels

### 6.3 Operator implementation strategy

- Selective `#pragma HLS BIND_OP ... impl=fabric`
- Reserve DSP usage for operations with stronger precision sensitivity

### 6.4 Memory/stream optimization

- LUTRAM-heavy buffering for intermediate arrays
- Reduced default stream depth (`SSMU_STREAM_DEPTH=16`)
- Deep FIFOs only on deadlock-sensitive streams

---

## 7) Validation and debug hooks

Available checkpoints used in this branch:

- `dut_ssm_core_f32.bin`
- `dut_ssm_normed_f32.bin`
- `dut_out_proj_f32.bin`
- `dut_out_f32.bin`

Reference files (from golden flow):

- `ssm_core_ref_f32.bin`
- `ssm_normed_ref_f32.bin`
- `out_proj_ref_f32.bin`
- `out_ref_f32.bin`

This enables stage-by-stage mismatch localization, not just final-output checking.

---

## 8) File map

- Kernel: `vitis5/quant9/ssm_lowrank_v3_mid5120_lr.cpp`
- Header: `vitis5/quant9/SSMU.h`
- Testbench: `vitis5/quant9/tb_ssm.cpp`
- Golden generator: `golden/make_bins_raw_lowrank.py`
- Bins: `vitis5/quant9/bins_raw`

---

## 9) TB workflow (what TB does, step by step)

The TB in `vitis5/quant9/tb_ssm.cpp` is a **bin-driven, golden-compare validator**.

### 9.1 Data loading stage

TB loads the following classes of data from `bins_raw`:

- Constant model parameters (`A_fixed`, `RMS_weight`, `RMS_weight_2`, `D_diag`)
- Projection weights (`W_in_1`, `W_in_2`, `W_in_nonlr`, `W_delta`, `W_out_A`, `W_out_B`)
- Runtime scales (`w_scale_in`, `w_scale_delta`, `w_scale_out`)
- Per-step IO (`kernel_in`, `x_in`, `out_ref`, `h1_ref`, optional `conv_state_out_ref`)
- Caches (`h0_in`, `conv_state_in`) with step override support

### 9.2 DUT invocation stage

TB calls `SSMU_STACK64` directly (matching `syn.top`) and captures:

- `out_dut`
- `h1_dut`
- `conv_state_out_dut`

If `H1` stream is disabled, TB falls back to `H1_ddr` to keep cache behavior valid.

### 9.3 Comparison stage

TB computes:

- `max_abs_diff(out_dut, out_ref)`
- `max_abs_diff(h1_dut, h1_ref)`
- `max_abs_diff(conv_state_out_dut, conv_state_out_ref)` (if ref exists)

Then compares against tolerance thresholds:

- `TB_TOL_OUT`
- `TB_TOL_H1`
- `TB_TOL_CS`

If any threshold is violated, TB reports top-K mismatches (token/lane/value).

### 9.4 Multi-step cache update stage

At the end of each step:

- Next `H0_in <- current H1_out`
- Next `conv_state_in <- current conv_state_out`

This emulates true autoregressive/cache-carry behavior across steps.

---

## 10) Test items checklist (what TB is validating)

Use this checklist when reporting test coverage.

### Functional correctness items

- [ ] Top function invocation correctness (`SSMU_STACK64` symbol is used)
- [ ] Input/weight file integrity and exact token count validation
- [ ] End-to-end output correctness (`out` vs `out_ref`)
- [ ] State update correctness (`h1` vs `h1_ref`)
- [ ] Convolution state correctness (`conv_state_out` vs reference, if provided)

### Numerical/quality items

- [ ] `max_abs_diff(out)` within `TB_TOL_OUT`
- [ ] `max_abs_diff(h1)` within `TB_TOL_H1`
- [ ] `max_abs_diff(conv_state_out)` within `TB_TOL_CS`
- [ ] Top-K mismatch report generated when failing

### Streaming/dataflow robustness items

- [ ] No stream underflow/overflow in TB drain checks
- [ ] Interleaved producer mode (`TB_INTERLEAVE=1`) runs cleanly
- [ ] Multi-step mode (`TB_STEPS>1`) preserves cache handoff consistency

### Deployment/readiness items

- [ ] Works in csim flow
- [ ] Works in cosim flow
- [ ] Bins auto-locate fallback works (or `TB_BINS_DIR` override works)

---

## 11) Common TB environment variables

| Variable | Purpose | Default |
|---|---|---|
| `TB_BINS_DIR` | Override bins directory | auto-locate `bins_raw` |
| `TB_TOL_OUT` | Out tolerance | `0.05` |
| `TB_TOL_H1` | H1 tolerance | `0.06` |
| `TB_TOL_CS` | Conv-state tolerance | `TB_TOL_H1` |
| `TB_STEPS` | Multi-step run count | `1` |
| `TB_TOPK` | Top mismatch lines to print | `10` |
| `TB_INTERLEAVE` | Deterministic interleaved input producer | `0` |
| `TB_PRINT_TOKEN` | Token index for detailed print | `0` |
| `TB_WSCALE_IN` | Override runtime input scale | from file/env logic |
| `TB_WSCALE_DELTA` | Override runtime delta scale | from file/env logic |
| `TB_WSCALE_OUT` | Override runtime output scale | from file/env logic |

---

