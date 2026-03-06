# ============================================================
# run_hls.tcl  –  Vitis HLS project script
# Usage:  vitis_hls -f run_hls.tcl
# ============================================================

set proj_name   mamba2_hls
set top_name    mamba2_forward
set part        xcu200-fsgd2104-2-e   ;# Alveo U250; change as needed

open_project   $proj_name
set_top        $top_name
open_solution  sol1 -flow_target vivado

# Target part & clock (5 ns = 200 MHz)
set_part       $part
create_clock -period 5 -name default

add_files      mamba2_hls.cpp -cflags "-I."
add_files -tb  mamba2_tb.cpp  -cflags "-I."

# C simulation
csim_design

# Synthesis only (no export – add export_design if you want IP)
csynth_design

close_project
