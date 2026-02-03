./sweep.py base code with the old Hadamard that was wrong 
./hadamard.py has the new hadamard implementation that quantizes the activations as well
./fixed.py has the fixed point code

## 03/02/2025
./out_proj_also.py most recent
./triton_stuff.py exploration of fused matmuls for speed up
result
```sh
bhardwaj@reactor:~/software/compression/fused$ uv run fused_lowrank_triton.py --fast --dtype fp16
max error vs FP32 ref: 1.499023e-01 (rel 4.110594e-04)
2xGEMM:      0.673 ms
Dense W_hat: 1.430 ms
Dense Full W:0.892 ms
Fused Triton:7.207 ms
```
