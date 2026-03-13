import numpy as np
import sys, os

FRAC_BITS = 12

def raw_i16_to_f32(path):
    a = np.fromfile(path, dtype=np.int16).astype(np.float32)
    return a / (2 ** FRAC_BITS)

def f32_from_bin(path):
    return np.fromfile(path, dtype=np.float32)

def stats(a, b, name):
    diff = a - b
    max_abs = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    idx = int(np.argmax(np.abs(diff)))
    print(f"[{name}] max_abs={max_abs:.6e}  rmse={rmse:.6e}  worst_idx={idx}  diff={diff[idx]:.6e}  ref={b[idx]:.6e}  dut={a[idx]:.6e}")
    return max_abs, rmse

def main():
    if len(sys.argv) != 4:
        print("Usage: py compare_outputs.py <bins_f32_dir> <bins_out_dir> <prefix>")
        print("  prefix example: out  (expects out_ref_f32.bin and out_dut.raw.bin)")
        sys.exit(1)

    bins_f32 = sys.argv[1]
    bins_out = sys.argv[2]
    prefix = sys.argv[3]

    out_ref = f32_from_bin(os.path.join(bins_f32, f"{prefix}_ref_f32.bin"))
    out_dut = raw_i16_to_f32(os.path.join(bins_out, f"{prefix}_dut.raw.bin"))

    if out_ref.size != out_dut.size:
        print(f"[ERR] size mismatch: ref={out_ref.size} dut={out_dut.size}")
        sys.exit(2)

    stats(out_dut, out_ref, prefix)

if __name__ == "__main__":
    main()
