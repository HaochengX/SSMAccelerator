
import os, csv, math, time
from dataclasses import dataclass
from typing import Dict, Tuple, Set, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

device = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if device == "cuda" else torch.float32

if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

def sync():
    if device == "cuda":
        torch.cuda.synchronize()

# ----------------------------
# Config
# ----------------------------
MODEL_ID = "state-spaces/mamba2-2.7b"
TOKENIZER_ID = "EleutherAI/gpt-neox-20b"

DATASET_ID = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
SPLIT = "test"

SEQ_LEN = 2048
BATCH_SIZE = 4
MAX_TOKENS = 200_000  # None for full split

IN_LAYER_MODES  = ["early_mid"]
OUT_LAYER_MODES = ["mid"]

IN_RANKS  = [None, 1280, 1024]                 # None = dense
OUT_RANKS = [None, 2176, 2048, 1920, 1792, 1536, 1280, 1024]     # rank for weight approximation; None=dense

CACHE_DIR = "./eval_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_IDS = os.path.join(CACHE_DIR, f"{DATASET_ID}_{DATASET_CONFIG}_{SPLIT}_neox_ids.pt")
CACHE_BLOCKS = os.path.join(CACHE_DIR, f"{DATASET_ID}_{DATASET_CONFIG}_{SPLIT}_blocks_{SEQ_LEN}_{MAX_TOKENS}.pt")

CSV_OUT = "sweep_sep_modes_inX_factor_outW_lowrank.csv"

# ----------------------------
# Factorized Linear (two GEMMs) for in_proj.x only
# ----------------------------
class FactorizedLinear(nn.Linear):
    """
    W(out,in) ≈ B(out,r) @ A(r,in)
    y = x @ A.T @ B.T + bias
    """
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.rank = 0
        self.register_buffer("A", None, persistent=False)  # (r, in)
        self.register_buffer("B", None, persistent=False)  # (out, r)

    @torch.no_grad()
    def disable_low_rank(self, free_buffers=True):
        self.rank = 0
        if free_buffers:
            self.A = None
            self.B = None

    @torch.no_grad()
    def set_low_rank_from_weight_gpu(self, rank: int):
        rmax = min(self.weight.shape)
        rank = min(rank, rmax)
        if rank >= rmax:
            self.disable_low_rank(True)
            return

        W = self.weight.detach().float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]

        A = Vh
        B = U * S.unsqueeze(0)

        dt = self.weight.dtype
        self.A = A.to(dtype=dt).contiguous()
        self.B = B.to(dtype=dt).contiguous()
        self.rank = rank

    def forward(self, x):
        if self.rank > 0 and self.A is not None and self.B is not None:
            y = x.matmul(self.A.transpose(0, 1))
            y = y.matmul(self.B.transpose(0, 1))
            if self.bias is not None:
                y = y + self.bias
            return y
        return F.linear(x, self.weight, self.bias)

def replace_linear_with_factorized(old: nn.Linear) -> FactorizedLinear:
    new = FactorizedLinear(
        old.in_features, old.out_features,
        bias=(old.bias is not None),
        device=old.weight.device, dtype=old.weight.dtype
    )
    new.weight.data.copy_(old.weight.data)
    if old.bias is not None:
        new.bias.data.copy_(old.bias.data)
    return new

# ----------------------------
# in_proj wrapper: keep z dense, factorize ONLY x, keep tail dense
# ----------------------------
class InProjXOnly(nn.Module):
    """
    in_proj outputs [z, x, B, C, dt] in Mamba2.
    Keep z dense, keep tail dense (B/C/dt), low-rank ONLY x.
    """
    def __init__(self, old: nn.Linear, z_dim: int, x_dim: int):
        super().__init__()
        assert isinstance(old, nn.Linear)
        assert 0 < z_dim and 0 < x_dim and (z_dim + x_dim) < old.out_features

        tail_dim = old.out_features - (z_dim + x_dim)

        self.z = nn.Linear(old.in_features, z_dim, bias=(old.bias is not None),
                           device=old.weight.device, dtype=old.weight.dtype)
        self.z.weight.data.copy_(old.weight.data[:z_dim])
        if old.bias is not None:
            self.z.bias.data.copy_(old.bias.data[:z_dim])

        x_lin = nn.Linear(old.in_features, x_dim, bias=(old.bias is not None),
                          device=old.weight.device, dtype=old.weight.dtype)
        x_lin.weight.data.copy_(old.weight.data[z_dim:z_dim + x_dim])
        if old.bias is not None:
            x_lin.bias.data.copy_(old.bias.data[z_dim:z_dim + x_dim])
        self.x = replace_linear_with_factorized(x_lin)

        self.tail = nn.Linear(old.in_features, tail_dim, bias=(old.bias is not None),
                              device=old.weight.device, dtype=old.weight.dtype)
        self.tail.weight.data.copy_(old.weight.data[z_dim + x_dim:])
        if old.bias is not None:
            self.tail.bias.data.copy_(old.bias.data[z_dim + x_dim:])

    def disable_low_rank(self):
        self.x.disable_low_rank(True)

    @torch.no_grad()
    def set_low_rank(self, rank: int):
        self.x.set_low_rank_from_weight_gpu(rank)

    def forward(self, x):
        return torch.cat([self.z(x), self.x(x), self.tail(x)], dim=-1)

# ----------------------------
# Layer selection
# ----------------------------
def layer_indices(mode: str, n_layers: int) -> Set[int]:
    if mode == "all":
        return set(range(n_layers))
    if mode == "early":
        return set(range(0, n_layers // 3))
    if mode == "mid":
        return set(range(n_layers // 3, 2 * n_layers // 3))
    if mode == "late":
        return set(range(2 * n_layers // 3, n_layers))
    if mode == "early_mid":
        return set(range(0, 2 * n_layers // 3))
    if mode == "mid_late":
        return set(range(n_layers // 3, n_layers))
    raise ValueError(mode)

# ----------------------------
# Dataset caching
# ----------------------------
def load_or_build_ids(tokenizer: AutoTokenizer) -> torch.Tensor:
    if os.path.exists(CACHE_IDS):
        return torch.load(CACHE_IDS, map_location="cpu")
    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split=SPLIT)
    text = "\n\n".join(t for t in ds["text"] if t and t.strip())
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0].contiguous()
    torch.save(ids, CACHE_IDS)
    return ids

def load_or_build_blocks(ids_1d: torch.Tensor) -> torch.Tensor:
    if os.path.exists(CACHE_BLOCKS):
        return torch.load(CACHE_BLOCKS, map_location="cpu")

    if MAX_TOKENS is not None and ids_1d.numel() > MAX_TOKENS:
        ids_1d = ids_1d[:MAX_TOKENS]

    n_tokens = ids_1d.numel()
    n_blocks = (n_tokens - 1) // SEQ_LEN
    usable = n_blocks * SEQ_LEN
    ids_1d = ids_1d[: usable + 1]

    blocks = ids_1d.unfold(0, SEQ_LEN + 1, SEQ_LEN).contiguous()
    torch.save(blocks, CACHE_BLOCKS)
    return blocks

# ----------------------------
# Patch model
# ----------------------------
def patch_inproj_xonly(model: nn.Module):
    m0 = model.backbone.layers[0].mixer
    d_inner = m0.out_proj.in_features  # 5120
    z_dim = d_inner
    x_dim = d_inner

    for layer in model.backbone.layers:
        mx = layer.mixer
        if not isinstance(mx.in_proj, InProjXOnly):
            mx.in_proj = InProjXOnly(mx.in_proj, z_dim=z_dim, x_dim=x_dim)

def disable_in_lowrank(model: nn.Module):
    for layer in model.backbone.layers:
        ip = layer.mixer.in_proj
        if isinstance(ip, InProjXOnly):
            ip.disable_low_rank()

# ----------------------------
# out_proj weight restore + cache
# ----------------------------
@dataclass
class OutW:
    W: torch.Tensor
    b: Optional[torch.Tensor]

OUT_W_BASE: Dict[int, OutW] = {}  # layer_idx -> original weight/bias (CPU fp16)

def snapshot_outproj_base(model: nn.Module):
    OUT_W_BASE.clear()
    for i, layer in enumerate(model.backbone.layers):
        op = layer.mixer.out_proj
        OUT_W_BASE[i] = OutW(
            W=op.weight.detach().to("cpu", dtype=torch.float16).clone(),
            b=None if op.bias is None else op.bias.detach().to("cpu", dtype=torch.float16).clone(),
        )

@torch.no_grad()
def restore_outproj_base(model: nn.Module):
    for i, layer in enumerate(model.backbone.layers):
        base = OUT_W_BASE[i]
        op = layer.mixer.out_proj
        op.weight.copy_(base.W.to(device=device, dtype=op.weight.dtype))
        if op.bias is not None and base.b is not None:
            op.bias.copy_(base.b.to(device=device, dtype=op.bias.dtype))

# cache low-rank reconstructed weights (CPU fp16)
OUT_W_CACHE: Dict[Tuple[int, int], torch.Tensor] = {}  # (layer_idx, rank) -> W_hat

@torch.no_grad()
def apply_outproj_weight_lowrank(model: nn.Module, rank: Optional[int], layers: Set[int], use_cache=True) -> float:
    """
    Replaces out_proj.weight with its rank-r approximation W_hat (dense weight).
    This WILL change PPL (because forward uses out_proj.weight via F.linear),
    but does NOT make compute low-rank (still dense GEMM).
    Returns total SVD time (seconds) (0 if served from cache).
    """
    if rank is None:
        return 0.0

    svd_time = 0.0
    for i, layer in enumerate(model.backbone.layers):
        if i not in layers:
            continue

        op = layer.mixer.out_proj
        W = op.weight
        rmax = min(W.shape)  # min(2560,5120)=2560
        r = min(rank, rmax)

        key = (i, r)
        if use_cache and key in OUT_W_CACHE:
            W_hat_cpu = OUT_W_CACHE[key]
            op.weight.copy_(W_hat_cpu.to(device=device, dtype=W.dtype, non_blocking=True))
            continue

        sync()
        t0 = time.time()
        Wf = W.detach().float()
        U, S, Vh = torch.linalg.svd(Wf, full_matrices=False)
        W_hat = (U[:, :r] * S[:r].unsqueeze(0)) @ Vh[:r, :]
        sync()
        svd_time += time.time() - t0

        W_hat_cpu = W_hat.to("cpu", dtype=torch.float16).contiguous()
        if use_cache:
            OUT_W_CACHE[key] = W_hat_cpu
        op.weight.copy_(W_hat_cpu.to(device=device, dtype=W.dtype, non_blocking=True))

    return svd_time

# ----------------------------
# Eval PPL
# ----------------------------
@torch.inference_mode()
def eval_ppl(model: nn.Module, blocks_cpu: torch.Tensor):
    total_nll = 0.0
    total_tok = 0
    n_blocks = blocks_cpu.size(0)

    if device == "cuda" and n_blocks > 0:
        warm = blocks_cpu[:1].to(device, non_blocking=True)
        _ = model(warm[:, :-1]).logits
        sync()

    t0 = time.time()
    pbar = tqdm(range(0, n_blocks, BATCH_SIZE), desc="Evaluating", unit="batch", leave=False)

    for i in pbar:
        batch = blocks_cpu[i:i+BATCH_SIZE].to(device, non_blocking=True)
        x = batch[:, :-1]
        y = batch[:, 1:]

        logits = model(x).logits
        if not torch.isfinite(logits).all():
            return float("inf"), float("inf"), total_tok, 0.0, 0.0

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="sum",
        )
        if not torch.isfinite(loss):
            return float("inf"), float("inf"), total_tok, 0.0, 0.0

        total_nll += loss.item()
        total_tok += y.numel()

        ppl_live = math.exp(total_nll / max(1, total_tok))
        tok_s_live = total_tok / max(1e-9, (time.time() - t0))
        pbar.set_postfix({"ppl": f"{ppl_live:.3f}", "tok/s": f"{tok_s_live:,.0f}"})

    sync()
    elapsed = time.time() - t0
    avg_nll = total_nll / total_tok
    ppl = math.exp(avg_nll)
    tok_s = total_tok / max(1e-9, elapsed)
    return ppl, avg_nll, total_tok, tok_s, elapsed

# ----------------------------
# AB cache for in_proj.x factorization
# ----------------------------
@dataclass
class AB:
    A: torch.Tensor
    B: torch.Tensor

AB_CACHE: Dict[Tuple[int, int], AB] = {}  # (layer_idx, rank) -> AB for in_proj.x

@torch.no_grad()
def enable_lowrank_in_x(model: nn.Module, rank: Optional[int], layers: Set[int], use_cache=True) -> float:
    if rank is None:
        return 0.0

    svd_time = 0.0
    for i, layer in enumerate(model.backbone.layers):
        if i not in layers:
            continue

        ip = layer.mixer.in_proj
        assert isinstance(ip, InProjXOnly)
        xproj = ip.x
        assert isinstance(xproj, FactorizedLinear)

        rmax = min(xproj.weight.shape)  # min(5120,2560)=2560
        if rank >= rmax:
            xproj.disable_low_rank(True)
            continue

        key = (i, rank)
        if use_cache and key in AB_CACHE:
            ab = AB_CACHE[key]
            xproj.A = ab.A.to(device=device, dtype=xproj.weight.dtype, non_blocking=True)
            xproj.B = ab.B.to(device=device, dtype=xproj.weight.dtype, non_blocking=True)
            xproj.rank = rank
            continue

        sync()
        t0 = time.time()
        xproj.set_low_rank_from_weight_gpu(rank)
        sync()
        svd_time += time.time() - t0

        if use_cache and xproj.A is not None and xproj.B is not None:
            AB_CACHE[key] = AB(
                A=xproj.A.detach().to("cpu", dtype=torch.float16),
                B=xproj.B.detach().to("cpu", dtype=torch.float16),
            )

    return svd_time

# ----------------------------
# Confirm out_proj bypass once
# ----------------------------
def assert_outproj_not_called(model: nn.Module):
    calls = {"out": 0}
    def hook(_m, _inp, _out): calls["out"] += 1
    h = model.backbone.layers[0].mixer.out_proj.register_forward_hook(hook)
    with torch.inference_mode():
        x = torch.randint(0, 100, (1, 16), device=device)
        _ = model(x).logits
    h.remove()
    print("out_proj forward hook calls:", calls["out"])
    # This being 0 is expected; if it becomes >0 in a future version, you can revisit factorized out_proj.
    return calls["out"]

# ----------------------------
# Main sweep
# ----------------------------
def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True)
    ids = load_or_build_ids(tokenizer)
    blocks = load_or_build_blocks(ids)
    if device == "cuda":
        blocks = blocks.pin_memory()

    print(f"Loading model: {MODEL_ID} ({DTYPE}) on {device}")
    model = MambaLMHeadModel.from_pretrained(MODEL_ID, device=device, dtype=DTYPE)
    model.eval()

    patch_inproj_xonly(model)
    snapshot_outproj_base(model)

    # confirm the out_proj path is bypassed (so we know why FactorizedLinear wouldn't work)
    assert_outproj_not_called(model)

    n_layers = len(model.backbone.layers)
    print(f"Detected layers: {n_layers}")

    # Baseline (dense)
    disable_in_lowrank(model)
    restore_outproj_base(model)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    base_ppl, base_nll, base_tok, base_tok_s, base_eval_s = eval_ppl(model, blocks)
    peak_gb = (torch.cuda.max_memory_allocated() / (1024**3)) if device == "cuda" else 0.0
    print(f"\nBASE: ppl={base_ppl:.4f}, tok/s={base_tok_s:,.0f}, eval={base_eval_s:.2f}s, peakVRAM={peak_gb:.2f} GB\n")

    # CSV
    new_file = (not os.path.exists(CSV_OUT)) or (os.stat(CSV_OUT).st_size == 0)
    with open(CSV_OUT, "a", newline="", buffering=1) as f:
        w = csv.writer(f)
        if new_file:
            w.writerow([
                "in_mode", "out_mode",
                "in_rank_x", "out_rank",
                "ppl", "delta_ppl",
                "svd_in_sec", "svd_outW_sec",
                "eval_sec", "tok_per_s",
                "peak_vram_gb"
            ])

        w.writerow(["BASE", "BASE", "DENSE", "DENSE",
                    f"{base_ppl:.6f}", "0.000000",
                    "0.00", "0.00",
                    f"{base_eval_s:.2f}", f"{base_tok_s:,.0f}", f"{peak_gb:.2f}"])
        f.flush()

        total = len(IN_LAYER_MODES) * len(OUT_LAYER_MODES) * len(IN_RANKS) * len(OUT_RANKS)
        sweep = tqdm(total=total, desc="Sweep", unit="cfg")

        for in_mode in IN_LAYER_MODES:
            in_layers = layer_indices(in_mode, n_layers)

            for out_mode in OUT_LAYER_MODES:
                out_layers = layer_indices(out_mode, n_layers)

                for in_rank in IN_RANKS:
                    for out_rank in OUT_RANKS:
                        # Reset to baseline weights each config
                        disable_in_lowrank(model)
                        restore_outproj_base(model)

                        if device == "cuda":
                            torch.cuda.reset_peak_memory_stats()

                        svd_in = enable_lowrank_in_x(model, in_rank, in_layers, use_cache=True)
                        svd_out = apply_outproj_weight_lowrank(model, out_rank, out_layers, use_cache=True)

                        ppl_val, avg_nll, total_tok, tok_s, eval_s = eval_ppl(model, blocks)
                        delta = ppl_val - base_ppl
                        peak_gb = (torch.cuda.max_memory_allocated() / (1024**3)) if device == "cuda" else 0.0

                        in_label = "DENSE" if in_rank is None else str(in_rank)
                        out_label = "DENSE" if out_rank is None else str(out_rank)

                        print(
                            f"in_mode={in_mode:9s} out_mode={out_mode:4s} "
                            f"inX={in_label:>5s} outW={out_label:>5s} "
                            f"ppl={ppl_val:.4f} Δ={delta:+.4f} "
                            f"svd(in)={svd_in:.2f}s svd(outW)={svd_out:.2f}s "
                            f"eval={eval_s:.2f}s tok/s={tok_s:,.0f} peakVRAM={peak_gb:.2f}GB"
                        )

                        w.writerow([
                            in_mode, out_mode,
                            in_label, out_label,
                            f"{ppl_val:.6f}", f"{delta:.6f}",
                            f"{svd_in:.2f}", f"{svd_out:.2f}",
                            f"{eval_s:.2f}", f"{tok_s:,.0f}",
                            f"{peak_gb:.2f}"
                        ])
                        f.flush()
                        sweep.update(1)

        sweep.close()

    print("\nDone. Results:", CSV_OUT)

if __name__ == "__main__":
    main()
