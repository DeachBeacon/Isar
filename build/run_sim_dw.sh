#!/bin/bash
# ============================================================================
# run_sim_dw.sh — compile + run dw_conv_engine sim (local Vivado xsim)
# ============================================================================
set -e
VIVADO_BIN="${VIVADO_BIN:-/d/vivado/2025.2/Vivado/bin}"
cd "$(dirname "$0")/../sim"

"$VIVADO_BIN/xvlog.bat" ../src/rtl/dw_conv_engine.v tb/tb_dw_conv_engine.v
"$VIVADO_BIN/xelab.bat" tb_dw_conv_engine -s tb_dw -debug typical
"$VIVADO_BIN/xsim.bat" tb_dw -R
