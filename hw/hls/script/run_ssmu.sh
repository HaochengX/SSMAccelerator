#!/bin/bash
# =============================================================================
# run_ssmu.sh — SSM_accel SSMU_STACK64 HLS runner (Vitis 2025.1)
# =============================================================================

# Load Vitis 2025.1 environment
if command -v vitis2025.1 &> /dev/null; then
    echo "[INFO] Loading Vitis 2025.1 environment using alias..."
    vitis2025.1
elif [ -f /tools/Xilinx/2025.1/Vitis/settings64.sh ]; then
    echo "[INFO] Loading Vitis 2025.1 from settings64.sh..."
    source /tools/Xilinx/2025.1/Vitis/settings64.sh
else
    echo "[WARNING] Could not find Vitis 2025.1 environment."
    echo "Please source it manually: source /tools/Xilinx/2025.1/Vitis/settings64.sh"
fi

# Check SSM_HOME
if [ -z "$SSM_HOME" ]; then
    echo "[ERROR] SSM_HOME environment variable is not set."
    echo "  export SSM_HOME=/home/chouy/SSM_accel"
    exit 1
fi

echo "========================================="
echo "  SSMU_STACK64 HLS Test Runner"
echo "========================================="
echo "SSM_HOME: $SSM_HOME"
echo ""

# ==================== CONFIGURATION ====================

PROJECT_NAME="ssmu_stack64_test"
TOP_FUNCTION="SSMU_STACK64"

# Source and testbench files (relative to SSM_HOME)
SOURCE_FILES=(
    "hw/hls/case/case_ssmu.cpp"
)
TB_FILES=(
    "hw/hls/case/case_ssmu.cpp"
)

# ==================== PARSE ARGUMENTS ====================

HLS_EXEC=${1:-1}  # Default: C simulation
SOLUTION_NAME=${2:-"solution1"}

echo "Execution mode: $HLS_EXEC"
case $HLS_EXEC in
    0) echo "  - Synthesis only" ;;
    1) echo "  - C simulation only" ;;
    2) echo "  - Synthesis + Co-simulation" ;;
    3) echo "  - Synthesis + IP Export (impl)" ;;
    4) echo "  - Full flow (impl)" ;;
    5) echo "  - Full flow (synth)" ;;
    6) echo "  - Synthesis + IP Export (syn only)" ;;
    7) echo "  - Synthesis + IP Export (syn + impl)" ;;
    8) echo "  - Synthesis + IP Export (no flow)" ;;
    9) echo "  - C-sim + Synthesis + IP Export (syn)" ;;
    *) echo "[WARNING] Unknown mode, defaulting to 1"
       HLS_EXEC=1 ;;
esac
echo "Solution: $SOLUTION_NAME"

# ==================== ENVIRONMENT VARIABLES ====================

# Bins directory — point to your bins_raw data
export TB_BINS_DIR="${SSM_HOME}/hw/hls/data/bins_raw"

# Tolerances (adjust as needed)
# export TB_TOL_OUT=0.05
# export TB_TOL_H1=0.06
# export TB_TOL_CS=0.06

# Multi-step inference
# export TB_STEPS=1

# Runtime weight scales (override if needed; else loaded from bins)
# export TB_WSCALE_IN=0.0
# export TB_WSCALE_DELTA=0.0
# export TB_WSCALE_OUT=0.0

# Debug prints
# export TB_TOPK=10
# export TB_PRINT_TOKEN=0
# export TB_INTERLEAVE=0

# ==================== BUILD ARGUMENTS ====================

SOURCE_ARGS=""
for src in "${SOURCE_FILES[@]}"; do
    SOURCE_ARGS="$SOURCE_ARGS --source_files $src"
done

TB_ARGS=""
for tb in "${TB_FILES[@]}"; do
    TB_ARGS="$TB_ARGS --tb_files $tb"
done

# ==================== RUN HLS ====================

echo ""
echo "Running HLS flow..."
echo "========================================="

cd "$SSM_HOME/hw/hls/script" || exit 1

python3 run_hls.py \
    --project_name "$PROJECT_NAME" \
    $SOURCE_ARGS \
    $TB_ARGS \
    --top_function "$TOP_FUNCTION" \
    --solution_name "$SOLUTION_NAME" \
    --part "xck26-sfvc784-2LV-c" \
    --clock_period 2.5 \
    --vivado_clock_period "4ns" \
    --hls_exec $HLS_EXEC \
    -p

echo ""
echo "========================================="
echo "HLS flow completed!"
echo "========================================="
echo ""
echo "Results:"
echo "  Project:  $SSM_HOME/hw/hls/syn_result/$PROJECT_NAME"
echo "  Logs:     $SSM_HOME/hw/hls/log/${PROJECT_NAME}_${SOLUTION_NAME}.log"
echo "  Reports:  $SSM_HOME/hw/hls/syn_result/$PROJECT_NAME/${SOLUTION_NAME}/syn/report/"
