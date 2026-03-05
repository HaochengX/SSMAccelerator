import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from tqdm import tqdm
from transformers import AutoTokenizer

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


@dataclass
class Config:
    model_id: str = "state-spaces/mamba2-2.7b"
    tokenizer_id: str = "EleutherAI/gpt-neox-20b"
    dataset_id: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    split: str = "test"
    seq_len: int = 2048
    batch_size: int = 2
    max_tokens: Optional[int] = 200_000
    cache_dir: str = "./eval_cache"
    csv_out: str = "mamba2_act_quant_bench.csv"
    run_modes: Tuple[str, ...] = (
        "fp16-fused",
        "fp16-unfused",
        "act8-naive",
        "act8-snc",
    )
    target_modules: Tuple[str, ...] = ("in_proj", "out_proj")
    calib_batches: int = 16
    act_bits: int = 8
    group_size: int = 64
    clip_percentile: Optional[float] = 0.9999


def cache_paths(cfg: Config) -> Tuple[str, str]:
    os.makedirs(cfg.cache_dir, exist_ok=True)
    ids_path = os.path.join(
        cfg.cache_dir,
        f"{cfg.dataset_id}_{cfg.dataset_config}_{cfg.split}_ids.pt",
    )
    blocks_path = os.path.join(
        cfg.cache_dir,
        f"{cfg.dataset_id}_{cfg.dataset_config}_{cfg.split}_blocks_{cfg.seq_len}_{cfg.max_tokens}.pt",
    )
    return ids_path, blocks_path


def load_or_build_ids(cfg: Config, tokenizer) -> torch.Tensor:
    ids_path, _ = cache_paths(cfg)
    if os.path.exists(ids_path):
        return torch.load(ids_path, map_location="cpu")

    ds = load_dataset(cfg.dataset_id, cfg.dataset_config, split=cfg.split)
    text = "\n\n".join(t for t in ds["text"] if t and t.strip())
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0].contiguous()
    torch.save(ids, ids_path)
    return ids


def load_or_build_blocks(cfg: Config, ids_1d: torch.Tensor) -> torch.Tensor:
    _, blocks_path = cache_paths(cfg)
    if os.path.exists(blocks_path):
        return torch.load(blocks_path, map_location="cpu")

    if cfg.max_tokens is not None and ids_1d.numel() > cfg.max_tokens:
        ids_1d = ids_1d[: cfg.max_tokens]

    n_blocks = (ids_1d.numel() - 1) // cfg.seq_len
    ids_1d = ids_1d[: n_blocks * cfg.seq_len + 1]
    blocks = ids_1d.unfold(0, cfg.seq_len + 1, cfg.seq_len).contiguous()
    torch.save(blocks, blocks_path)
    return blocks


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
        batch = blocks_cpu[i : i + cfg.batch_size].to(device, non_blocking=True)
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
        pbar.set_postfix({"ppl": f"{math.exp(total_nll / max(1, total_tok)):.3f}"})

    sync(device)
    elapsed = time.time() - t0
    avg_nll = total_nll / total_tok
    ppl = math.exp(avg_nll)
    tok_s = total_tok / max(elapsed, 1e-9)
    return ppl, avg_nll, total_tok, tok_s, elapsed


def open_csv(path: str):
    is_new = (not os.path.exists(path)) or (os.stat(path).st_size == 0)
    f = open(path, "a", newline="", buffering=1)
    writer = csv.writer(f)
    if is_new:
        writer.writerow(
            [
                "run_mode",
                "target_modules",
                "act_bits",
                "group_size",
                "clip_percentile",
                "ppl",
                "delta_ppl_vs_fp16_fused",
                "eval_sec",
                "tok_per_s",
                "peak_vram_gb",
            ]
        )
    return f, writer


class NaiveActQuant(nn.Module):
    def __init__(self, n_bits: int = 8, clip_percentile: Optional[float] = 0.9999, eps: float = 1e-8):
        super().__init__()
        self.n_bits = n_bits
        self.clip_percentile = clip_percentile
        self.eps = eps
        self.calibrating = False
        self.samples = []
        self.register_buffer("scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("ready", torch.tensor(False))

    @torch.no_grad()
    def start_calibration(self):
        self.calibrating = True
        self.samples = []
        self.ready.fill_(False)

    @torch.no_grad()
    def finish_calibration(self):
        vals = torch.cat(self.samples, dim=0) if self.samples else torch.tensor([1.0], dtype=torch.float32)
        a = vals.abs()
        if self.clip_percentile is None:
            amax = a.max()
        else:
            amax = torch.quantile(a, self.clip_percentile)
        qmax = (1 << (self.n_bits - 1)) - 1
        self.scale.copy_(torch.clamp(amax / qmax, min=self.eps))
        self.samples = []
        self.calibrating = False
        self.ready.fill_(True)

    @torch.no_grad()
    def _collect(self, x: torch.Tensor):
        flat = x.detach().float().reshape(-1)
        if flat.numel() > 65536:
            idx = torch.randperm(flat.numel(), device=flat.device)[:65536]
            flat = flat[idx]
        self.samples.append(flat.cpu())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.calibrating:
            self._collect(x)
            return x
        if not bool(self.ready):
            return x
        qmax = (1 << (self.n_bits - 1)) - 1
        s = self.scale.to(device=x.device, dtype=x.dtype)
        q = torch.clamp(torch.round(x / s), -qmax, qmax)
        return q * s


class SortClusterActQuant(nn.Module):
    """
    Quamba2-like approximation:
    - calibrate per-channel maxima on the last dim
    - sort channels by calibrated magnitude
    - cluster adjacent sorted channels into groups
    - one symmetric int8 scale per group
    - quantize in sorted space
    - unsort back
    """
    def __init__(self, n_bits: int = 8, group_size: int = 64, clip_percentile: Optional[float] = 0.9999, eps: float = 1e-8):
        super().__init__()
        self.n_bits = n_bits
        self.group_size = group_size
        self.clip_percentile = clip_percentile
        self.eps = eps
        self.calibrating = False
        self.channel_samples = []
        self.register_buffer("perm", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("inv_perm", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("group_scales", torch.empty(0, dtype=torch.float32), persistent=False)
        self.register_buffer("ready", torch.tensor(False))

    @torch.no_grad()
    def start_calibration(self):
        self.calibrating = True
        self.channel_samples = []
        self.ready.fill_(False)

    @torch.no_grad()
    def finish_calibration(self):
        if not self.channel_samples:
            self.perm = torch.empty(0, dtype=torch.long)
            self.inv_perm = torch.empty(0, dtype=torch.long)
            self.group_scales = torch.empty(0, dtype=torch.float32)
            self.calibrating = False
            self.ready.fill_(False)
            return

        ch_max = torch.stack(self.channel_samples, dim=0).max(dim=0).values
        perm = torch.argsort(ch_max, descending=True)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.numel(), dtype=perm.dtype)

        sorted_max = ch_max[perm]
        qmax = (1 << (self.n_bits - 1)) - 1

        scales = []
        n = sorted_max.numel()
        g = self.group_size if self.group_size > 0 else n
        for s in range(0, n, g):
            block = sorted_max[s : s + g]
            if self.clip_percentile is None:
                amax = block.max()
            else:
                # sorted_max is already a condensed per-channel stat; use max-in-group here
                amax = block.max()
            scale = torch.clamp(amax / qmax, min=self.eps)
            scales.append(scale)

        self.perm = perm.cpu()
        self.inv_perm = inv_perm.cpu()
        self.group_scales = torch.stack(scales).cpu()
        self.channel_samples = []
        self.calibrating = False
        self.ready.fill_(True)

    @torch.no_grad()
    def _collect(self, x: torch.Tensor):
        # x shape [..., C], collect channel-wise max over all leading dims
        ch_max = x.detach().float().abs().reshape(-1, x.shape[-1]).max(dim=0).values
        self.channel_samples.append(ch_max.cpu())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.calibrating:
            self._collect(x)
            return x

        if not bool(self.ready) or self.perm.numel() == 0:
            return x

        perm = self.perm.to(device=x.device)
        inv_perm = self.inv_perm.to(device=x.device)
        scales = self.group_scales.to(device=x.device, dtype=x.dtype)

        xs = x.index_select(dim=-1, index=perm)
        qmax = (1 << (self.n_bits - 1)) - 1
        g = self.group_size if self.group_size > 0 else xs.shape[-1]

        chunks = []
        n = xs.shape[-1]
        for gi, s in enumerate(range(0, n, g)):
            block = xs[..., s : s + g]
            scale = scales[gi]
            q = torch.clamp(torch.round(block / scale), -qmax, qmax)
            chunks.append(q * scale)

        ys = torch.cat(chunks, dim=-1)
        y = ys.index_select(dim=-1, index=inv_perm)
        return y


class QuantWrappedLinear(nn.Module):
    def __init__(self, linear: nn.Module, act_in_q: Optional[nn.Module] = None):
        super().__init__()
        self.linear = linear
        self.act_in_q = act_in_q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_in_q is not None:
            x = self.act_in_q(x)
        return self.linear(x)


def disable_mamba2_mem_eff_path(model: nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "use_mem_eff_path"):
            module.use_mem_eff_path = False


def build_model(model_id: str, device: str) -> nn.Module:
    model = MambaLMHeadModel.from_pretrained(
        model_id,
        device="cpu",
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.eval()
    return model


def collect_act_quant_modules(model: nn.Module) -> List[nn.Module]:
    out = []
    for m in model.modules():
        if isinstance(m, (NaiveActQuant, SortClusterActQuant)):
            out.append(m)
    return out


@torch.inference_mode()
def calibrate_activation_quantizers(model: nn.Module, blocks_cpu: torch.Tensor, cfg: Config, device: str) -> None:
    qs = collect_act_quant_modules(model)
    if not qs:
        return

    for q in qs:
        q.start_calibration()

    n_blocks = min(blocks_cpu.size(0), cfg.calib_batches * cfg.batch_size)
    for i in range(0, n_blocks, cfg.batch_size):
        batch = blocks_cpu[i : i + cfg.batch_size].to(device, non_blocking=True)
        x = batch[:, :-1]
        _ = model(x).logits

    sync(device)

    for q in qs:
        q.finish_calibration()


def apply_activation_quant_only(
    model: nn.Module,
    run_mode: str,
    target_modules: Tuple[str, ...],
    cfg: Config,
    device: str,
) -> nn.Module:
    for layer in model.backbone.layers:
        mixer = layer.mixer

        if "in_proj" in target_modules:
            old = mixer.in_proj
            if not isinstance(old, nn.Linear):
                raise TypeError(f"Expected nn.Linear for in_proj, got {type(old)}")
            if run_mode == "act8-naive":
                q = NaiveActQuant(n_bits=cfg.act_bits, clip_percentile=cfg.clip_percentile)
            elif run_mode == "act8-snc":
                q = SortClusterActQuant(
                    n_bits=cfg.act_bits,
                    group_size=cfg.group_size,
                    clip_percentile=cfg.clip_percentile,
                )
            else:
                q = None
            mixer.in_proj = QuantWrappedLinear(old, act_in_q=q)

        if "out_proj" in target_modules:
            old = mixer.out_proj
            if not isinstance(old, nn.Linear):
                raise TypeError(f"Expected nn.Linear for out_proj, got {type(old)}")
            if run_mode == "act8-naive":
                q = NaiveActQuant(n_bits=cfg.act_bits, clip_percentile=cfg.clip_percentile)
            elif run_mode == "act8-snc":
                q = SortClusterActQuant(
                    n_bits=cfg.act_bits,
                    group_size=cfg.group_size,
                    clip_percentile=cfg.clip_percentile,
                )
            else:
                q = None
            mixer.out_proj = QuantWrappedLinear(old, act_in_q=q)

    model.to(device)
    return model


def iter_runs(cfg: Config):
    for run_mode in cfg.run_modes:
        yield run_mode, cfg.target_modules


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Activation-only Mamba2 benchmark with Quamba2-like sort-and-cluster fake quant.")
    p.add_argument("--model-id", default="state-spaces/mamba2-2.7b")
    p.add_argument("--tokenizer-id", default="EleutherAI/gpt-neox-20b")
    p.add_argument("--csv-out", default="mamba2_act_quant_bench.csv")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=200_000)
    p.add_argument(
        "--run-modes",
        nargs="+",
        default=["fp16-fused", "fp16-unfused", "act8-naive", "act8-snc"],
        choices=["fp16-fused", "fp16-unfused", "act8-naive", "act8-snc"],
    )
    p.add_argument(
        "--target-modules",
        nargs="+",
        default=["in_proj", "out_proj"],
        choices=["in_proj", "out_proj"],
    )
    p.add_argument("--act-bits", type=int, default=8)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--clip-percentile", type=float, default=0.9999)
    p.add_argument("--calib-batches", type=int, default=16)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        model_id=args.model_id,
        tokenizer_id=args.tokenizer_id,
        csv_out=args.csv_out,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        run_modes=tuple(args.run_modes),
        target_modules=tuple(args.target_modules),
        act_bits=args.act_bits,
        group_size=args.group_size,
        clip_percentile=args.clip_percentile,
        calib_batches=args.calib_batches,
    )

    runs = list(iter_runs(cfg))
    print(f"Planned runs: {len(runs)}")
    for run_mode, target_modules in runs:
        print(f"  run_mode={run_mode:>12s} targets={','.join(target_modules)}")

    if args.dry_run:
        return

    device = get_device()
    if device != "cuda":
        raise RuntimeError("This script requires CUDA.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_id, use_fast=True)
    ids = load_or_build_ids(cfg, tokenizer)
    blocks = load_or_build_blocks(cfg, ids).pin_memory()

    f, writer = open_csv(cfg.csv_out)
    base_ppl = None

    try:
        for run_mode, target_modules in runs:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            print(f"\nLoading model for run: run_mode={run_mode}, targets={target_modules}")
            model = build_model(cfg.model_id, device="cpu")

            if run_mode == "fp16-fused":
                model.to(device)

            elif run_mode == "fp16-unfused":
                model.to(device)
                disable_mamba2_mem_eff_path(model)

            else:
                model.to(device)
                disable_mamba2_mem_eff_path(model)
                model = apply_activation_quant_only(
                    model=model,
                    run_mode=run_mode,
                    target_modules=target_modules,
                    cfg=cfg,
                    device=device,
                )
                calibrate_activation_quantizers(model, blocks, cfg, device)

            model.eval()

            ppl, _, _, tok_s, eval_s = eval_ppl(model, blocks, cfg, device)
            peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

            if run_mode == "fp16-fused":
                base_ppl = ppl
            if base_ppl is None:
                base_ppl = ppl
            delta = ppl - base_ppl

            target_label = ",".join(target_modules)
            print(
                f"run_mode={run_mode:>12s} targets={target_label:>15s} "
                f"ppl={ppl:.4f} delta={delta:+.4f} eval={eval_s:.2f}s "
                f"tok/s={tok_s:,.0f} peak={peak_gb:.2f}GB"
            )

            writer.writerow(
                [
                    run_mode,
                    target_label,
                    str(cfg.act_bits),
                    str(cfg.group_size),
                    str(cfg.clip_percentile),
                    f"{ppl:.6f}",
                    f"{delta:.6f}",
                    f"{eval_s:.2f}",
                    f"{tok_s:,.0f}",
                    f"{peak_gb:.2f}",
                ]
            )
            f.flush()

            del model
            torch.cuda.empty_cache()

    finally:
        f.close()

    print(f"\nDone. Results written to {cfg.csv_out}")


if __name__ == "__main__":
    main()
