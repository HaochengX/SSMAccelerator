import os, csv, math, time
from dataclasses import dataclass
from typing import Dict, Tuple, Set, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_dtype(device: str):
    return torch.float16 if device == "cuda" else torch.float32

def sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()

# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    model_id: str = "state-spaces/mamba2-2.7b"
    tokenizer_id: str = "EleutherAI/gpt-neox-20b"

    dataset_id: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    split: str = "test"

    seq_len: int = 2048
    batch_size: int = 4
    max_tokens: Optional[int] = 200_000

    in_layer_modes: Tuple[str, ...] = ("early_mid",)
    out_layer_modes: Tuple[str, ...] = ("mid",)

    in_ranks: Tuple[Optional[int], ...] = (None, 1280, 1024)
    out_ranks: Tuple[Optional[int], ...] = (None, 1792, 1536)

    cache_dir: str = "./eval_cache"
    csv_out: str = "sweep_sep_modes_inX_factor_outW_lowrank.csv"
    force_unfused: bool = False

# ----------------------------
# Factorized Linear (two GEMMs)
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
# Low-rank factor cache
# ----------------------------
@dataclass
class AB:
    A: torch.Tensor
    B: torch.Tensor

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
def build_cache_paths(cfg: Config):
    os.makedirs(cfg.cache_dir, exist_ok=True)
    ids_path = os.path.join(cfg.cache_dir, f"{cfg.dataset_id}_{cfg.dataset_config}_{cfg.split}_neox_ids.pt")
    blocks_path = os.path.join(cfg.cache_dir, f"{cfg.dataset_id}_{cfg.dataset_config}_{cfg.split}_blocks_{cfg.seq_len}_{cfg.max_tokens}.pt")
    return ids_path, blocks_path

def load_or_build_ids(cfg: Config, tokenizer: AutoTokenizer) -> torch.Tensor:
    ids_path, _ = build_cache_paths(cfg)
    if os.path.exists(ids_path):
        return torch.load(ids_path, map_location="cpu")
    ds = load_dataset(cfg.dataset_id, cfg.dataset_config, split=cfg.split)
    text = "\n\n".join(t for t in ds["text"] if t and t.strip())
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0].contiguous()
    torch.save(ids, ids_path)
    return ids

def load_or_build_blocks(cfg: Config, ids_1d: torch.Tensor) -> torch.Tensor:
    _, blocks_path = build_cache_paths(cfg)
    if os.path.exists(blocks_path):
        return torch.load(blocks_path, map_location="cpu")

    if cfg.max_tokens is not None and ids_1d.numel() > cfg.max_tokens:
        ids_1d = ids_1d[:cfg.max_tokens]

    n_tokens = ids_1d.numel()
    n_blocks = (n_tokens - 1) // cfg.seq_len
    usable = n_blocks * cfg.seq_len
    ids_1d = ids_1d[: usable + 1]

    blocks = ids_1d.unfold(0, cfg.seq_len + 1, cfg.seq_len).contiguous()
    torch.save(blocks, blocks_path)
    return blocks

# ----------------------------
# Patching helpers
# ----------------------------
def patch_inproj_xonly(model: nn.Module):
    m0 = model.backbone.layers[0].mixer
    d_inner = m0.out_proj.in_features
    z_dim = d_inner
    x_dim = d_inner

    for layer in model.backbone.layers:
        mx = layer.mixer
        if not isinstance(mx.in_proj, InProjXOnly):
            mx.in_proj = InProjXOnly(mx.in_proj, z_dim=z_dim, x_dim=x_dim)

def patch_outproj_factorized(model: nn.Module):
    for layer in model.backbone.layers:
        op = layer.mixer.out_proj
        if isinstance(op, nn.Linear) and not isinstance(op, FactorizedLinear):
            layer.mixer.out_proj = replace_linear_with_factorized(op)

def set_force_unfused(model: nn.Module, enabled: bool):
    for layer in model.backbone.layers:
        mixer = layer.mixer
        if hasattr(mixer, "use_mem_eff_path"):
            mixer.use_mem_eff_path = not enabled

# ----------------------------
# Out-proj baseline snapshot
# ----------------------------
@dataclass
class OutW:
    W: torch.Tensor
    b: Optional[torch.Tensor]

OUT_W_BASE: Dict[int, OutW] = {}

def snapshot_outproj_base(model: nn.Module):
    OUT_W_BASE.clear()
    for i, layer in enumerate(model.backbone.layers):
        op = layer.mixer.out_proj
        OUT_W_BASE[i] = OutW(
            W=op.weight.detach().to("cpu", dtype=torch.float16).clone(),
            b=None if op.bias is None else op.bias.detach().to("cpu", dtype=torch.float16).clone(),
        )

@torch.no_grad()
def restore_outproj_base(model: nn.Module, device: str):
    for i, layer in enumerate(model.backbone.layers):
        base = OUT_W_BASE[i]
        op = layer.mixer.out_proj
        op.weight.copy_(base.W.to(device=device, dtype=op.weight.dtype))
        if op.bias is not None and base.b is not None:
            op.bias.copy_(base.b.to(device=device, dtype=op.bias.dtype))
        if isinstance(op, FactorizedLinear):
            op.disable_low_rank(True)

# ----------------------------
# Low-rank SVD + caching
# ----------------------------
def apply_lowrank_factorized(
    op: FactorizedLinear,
    rank: int,
    cache: Dict[Tuple[int, int], AB],
    key_prefix: int,
    use_cache: bool,
    device: str,
) -> float:
    rmax = min(op.weight.shape)
    r = min(rank, rmax)
    if r >= rmax:
        op.disable_low_rank(True)
        return 0.0

    key = (key_prefix, r)
    if use_cache and key in cache:
        ab = cache[key]
        op.A = ab.A.to(device=device, dtype=op.weight.dtype, non_blocking=True)
        op.B = ab.B.to(device=device, dtype=op.weight.dtype, non_blocking=True)
        op.rank = r
        return 0.0

    sync(device)
    t0 = time.time()
    op.set_low_rank_from_weight_gpu(r)
    sync(device)
    svd_time = time.time() - t0

    if use_cache and op.A is not None and op.B is not None:
        cache[key] = AB(
            A=op.A.detach().to("cpu", dtype=torch.float16),
            B=op.B.detach().to("cpu", dtype=torch.float16),
        )

    return svd_time

AB_CACHE: Dict[Tuple[int, int], AB] = {}
OUT_AB_CACHE: Dict[Tuple[int, int], AB] = {}

@torch.no_grad()
def enable_lowrank_in_x(model: nn.Module, rank: Optional[int], layers: Set[int], use_cache: bool, device: str) -> float:
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

        svd_time += apply_lowrank_factorized(
            xproj,
            rank=rank,
            cache=AB_CACHE,
            key_prefix=i,
            use_cache=use_cache,
            device=device,
        )

    return svd_time

@torch.no_grad()
def enable_lowrank_outproj(model: nn.Module, rank: Optional[int], layers: Set[int], use_cache: bool, device: str) -> float:
    if rank is None:
        return 0.0

    svd_time = 0.0
    for i, layer in enumerate(model.backbone.layers):
        if i not in layers:
            continue
        op = layer.mixer.out_proj
        assert isinstance(op, FactorizedLinear)

        svd_time += apply_lowrank_factorized(
            op,
            rank=rank,
            cache=OUT_AB_CACHE,
            key_prefix=i,
            use_cache=use_cache,
            device=device,
        )

    return svd_time

# ----------------------------
# Eval PPL
# ----------------------------
@torch.inference_mode()
def eval_ppl(model: nn.Module, blocks_cpu: torch.Tensor, cfg: Config, device: str):
    total_nll = 0.0
    total_tok = 0
    n_blocks = blocks_cpu.size(0)

    if device == "cuda" and n_blocks > 0:
        warm = blocks_cpu[:1].to(device, non_blocking=True)
        _ = model(warm[:, :-1]).logits
        sync(device)

    t0 = time.time()
    pbar = tqdm(range(0, n_blocks, cfg.batch_size), desc="Evaluating", unit="batch", leave=False)

    for i in pbar:
        batch = blocks_cpu[i:i+cfg.batch_size].to(device, non_blocking=True)
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

    sync(device)
    elapsed = time.time() - t0
    avg_nll = total_nll / total_tok
    ppl = math.exp(avg_nll)
    tok_s = total_tok / max(1e-9, elapsed)
    return ppl, avg_nll, total_tok, tok_s, elapsed

# ----------------------------
# Main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--force-unfused", action="store_true", help="Disable mem-eff fused path so out_proj.forward is called")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = Config(force_unfused=bool(args.force_unfused))
    device = get_device()
    dtype = get_dtype(device)

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_id, use_fast=True)
    ids = load_or_build_ids(cfg, tokenizer)
    blocks = load_or_build_blocks(cfg, ids)
    if device == "cuda":
        blocks = blocks.pin_memory()

    print(f"Loading model: {cfg.model_id} ({dtype}) on {device}")
    model = MambaLMHeadModel.from_pretrained(cfg.model_id, device=device, dtype=dtype)
    model.eval()

    # Patch and snapshot
    patch_inproj_xonly(model)
    patch_outproj_factorized(model)
    if cfg.force_unfused:
        set_force_unfused(model, enabled=True)
        print("Force unfused: use_mem_eff_path disabled")
    snapshot_outproj_base(model)

    n_layers = len(model.backbone.layers)
    print(f"Detected layers: {n_layers}")

    # Baseline (dense)
    restore_outproj_base(model, device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    base_ppl, _base_nll, _base_tok, base_tok_s, base_eval_s = eval_ppl(model, blocks, cfg, device)
    peak_gb = (torch.cuda.max_memory_allocated() / (1024**3)) if device == "cuda" else 0.0
    print(f"\nBASE: ppl={base_ppl:.4f}, tok/s={base_tok_s:,.0f}, eval={base_eval_s:.2f}s, peakVRAM={peak_gb:.2f} GB\n")

    # CSV
    new_file = (not os.path.exists(cfg.csv_out)) or (os.stat(cfg.csv_out).st_size == 0)
    with open(cfg.csv_out, "a", newline="", buffering=1) as f:
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

        total = len(cfg.in_layer_modes) * len(cfg.out_layer_modes) * len(cfg.in_ranks) * len(cfg.out_ranks)
        sweep = tqdm(total=total, desc="Sweep", unit="cfg")

        for in_mode in cfg.in_layer_modes:
            in_layers = layer_indices(in_mode, n_layers)

            for out_mode in cfg.out_layer_modes:
                out_layers = layer_indices(out_mode, n_layers)

                for in_rank in cfg.in_ranks:
                    for out_rank in cfg.out_ranks:
                        restore_outproj_base(model, device)
                        if device == "cuda":
                            torch.cuda.reset_peak_memory_stats()

                        svd_in = enable_lowrank_in_x(model, in_rank, in_layers, use_cache=True, device=device)

                        svd_out = enable_lowrank_outproj(model, out_rank, out_layers, use_cache=True, device=device)

                        ppl_val, _avg_nll, _total_tok, tok_s, eval_s = eval_ppl(model, blocks, cfg, device)
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

    print("\nDone. Results:", cfg.csv_out)

if __name__ == "__main__":
    main()
