import csv
import itertools
import os
import time
import torch
import torch.nn.functional as F
import gc
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Enable better memory management for fragmented GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_ID = "state-spaces/mamba-2.8b-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- Utils ----------------
def apply_low_rank_(w, rank):
    if rank >= min(w.shape):
        return
    U, S, Vh = torch.linalg.svd(w.float(), full_matrices=False)
    w.copy_((U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank]).to(w.dtype))


def fake_quant_(w, bits, group_size):
    qmax = 2 ** (bits - 1) - 1
    wv = w.view(-1, group_size)
    scale = wv.abs().amax(dim=-1, keepdim=True) / qmax
    scale.clamp_(min=1e-8)
    w.copy_(((wv / scale).round().clamp(-qmax, qmax) * scale).view_as(w))


def hadamard_quant_(w: torch.Tensor, bits=4, group_size=64):
    orig_dtype = w.dtype
    n = w.shape[-1]
    next_pow2 = 2 ** ((n - 1).bit_length())
    w_padded = F.pad(w.float(), (0, next_pow2 - n))

    H = torch.tensor([[1.0]], device=w.device)
    while H.shape[0] < next_pow2:
        H = torch.cat((torch.cat((H, H), dim=1), torch.cat((H, -H), dim=1)), dim=0)
    H = H / torch.sqrt(torch.tensor(next_pow2, device=w.device))

    w_rotated = w_padded @ H
    orig_rotated_shape = w_rotated.shape
    w_flat = w_rotated.reshape(-1, group_size)
    q_max = 2 ** (bits - 1) - 1
    scale = w_flat.abs().max(dim=-1, keepdim=True)[0] / q_max
    scale = scale.clamp(min=1e-8)

    w_quant = (w_flat / scale).round().clamp(-q_max, q_max) * scale
    w_rotated = w_quant.reshape(orig_rotated_shape)
    w_out = (w_rotated @ H)[..., :n]
    return w_out.to(orig_dtype)


def layer_indices(mode, n_layers):
    if mode == "all":
        return set(range(n_layers))
    if mode == "early":
        return set(range(2, n_layers // 3))
    if mode == "mid":
        return set(range(n_layers // 3, 2 * n_layers // 3))
    if mode == "late":
        return set(range(2 * n_layers // 3, n_layers))
    raise ValueError(mode)


def compress_mamba(model, rb, rc, rdt, bits, gsize, layers, quant_type):
    for i, layer in enumerate(model.backbone.layers):
        if i not in layers:
            continue
        m = layer.mixer
        DT = m.dt_proj.in_features
        DS = m.config.state_size
        W = m.x_proj.weight.data

        _, wb, wc = torch.split(W, [DT, DS, DS], dim=0)

        apply_low_rank_(wb, rb)
        apply_low_rank_(wc, rc)
        apply_low_rank_(m.dt_proj.weight.data, rdt)

        target_weights = [wb, wc, m.dt_proj.weight.data]
        for tw in target_weights:
            if quant_type == "fake":
                fake_quant_(tw, bits, gsize)
            elif quant_type == "hadamard":
                tw.copy_(hadamard_quant_(tw, bits, gsize))


def evaluate(model, ids):
    model.eval()
    nlls = []
    stride, seq_len = 512, 1024
    with torch.inference_mode():
        for i in range(0, min(ids.size(1), 5000), stride):
            end = min(i + seq_len, ids.size(1))
            if end < seq_len:
                continue
            x = ids[:, end - seq_len : end].to(DEVICE)
            y = x.clone()
            y[:, : -(end - i)] = -100
            logits = model(x).logits
            loss = F.cross_entropy(
                logits[..., :-1, :].reshape(-1, logits.size(-1)), y[..., 1:].reshape(-1)
            )
            nlls.append(loss)
    return torch.exp(torch.stack(nlls).mean()).item()


# ---------------- Main ----------------


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tokenizer(
        "\n\n".join(t for t in dataset["text"] if t.strip()), return_tensors="pt"
    ).input_ids

    # 1. Load the "Clean" model to CPU once
    print(f"--- Initializing: Loading {MODEL_ID} to CPU RAM ---")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    # Store the clean state locally on CPU
    clean_state = {k: v.clone().cpu() for k, v in base_model.state_dict().items()}

    # Sweep Parameters
    ranks_b = [16, 12, 8, 4]
    ranks_c = [16, 12, 8, 4]
    ranks_dt = [160, 120, 80, 40, 20]
    bits_list = [8, 4]
    group_sizes = [64, 128, 256]
    layer_modes = ["all", "mid", "late", "early"]
    quant_types = ["hadamard", "fake"]

    total = (
        len(bits_list)
        * len(group_sizes)
        * len(ranks_b)
        * len(ranks_c)
        * len(ranks_dt)
        * len(layer_modes)
        * len(quant_types)
    )
    csv_file = "hadamard_all_sweep_2.8b_optimized.csv"

    with open(csv_file, "a", newline="", buffering=1) as f:
        writer = csv.writer(f)
        if not os.path.exists(csv_file) or os.stat(csv_file).st_size == 0:
            writer.writerow(
                ["quant", "bits", "group", "rb", "rc", "rdt", "layer_mode", "ppl"]
            )

        for qt, b, g, rb, rc, rdt, lm in tqdm(
            itertools.product(
                quant_types,
                bits_list,
                group_sizes,
                ranks_b,
                ranks_c,
                ranks_dt,
                layer_modes,
            ),
            total=total,
        ):
            # 2. Reset model weights from clean CPU state
            base_model.load_state_dict(clean_state)
            model = base_model.to(DEVICE)

            layers = layer_indices(lm, len(model.backbone.layers))
            compress_mamba(model, rb, rc, rdt, b, g, layers, qt)

            ppl = evaluate(model, ids)
            writer.writerow([qt, b, g, rb, rc, rdt, lm, f"{ppl:.4f}"])
            f.flush()

            # 3. Aggressive Cleanup
            model.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    main()
