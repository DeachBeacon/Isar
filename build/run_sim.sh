#!/bin/bash
# ============================================================================
# run_sim.sh — compile + elaborate + run pw_conv_engine sim with local Vivado xsim
# Usage: bash build/run_sim.sh
# Note : all config comes from sim/data/params.hex (written by gen_golden.py);
#        no command-line plusargs, because Git Bash mangles "=" into .bat files.
# ============================================================================
set -e
VIVADO_BIN="${VIVADO_BIN:-/d/vivado/2025.2/Vivado/bin}"
cd "$(dirname "$0")/../sim"

"$VIVADO_BIN/xvlog.bat" ../src/rtl/pw_conv_engine.v tb/tb_pw_conv_engine.v
"$VIVADO_BIN/xelab.bat" tb_pw_conv_engine -s tb_pw -debug typical
"$VIVADO_BIN/xsim.bat" tb_pw -R
