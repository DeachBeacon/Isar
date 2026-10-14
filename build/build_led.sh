#!/bin/bash
# ============================================================================
# build_led.sh — headless 生成 LED 点灯 bitstream
# 用法：bash build/build_led.sh
# 注意：vivado 批处理在含空格的仓库路径下运行，故在仓库根用绝对路径引用文件。
# ============================================================================
set -e
VIVADO_BIN="${VIVADO_BIN:-/d/vivado/2025.2/Vivado/bin}"
cd "$(dirname "$0")/.."          # 回到仓库根

"$VIVADO_BIN/vivado.bat" -mode batch -source build/build_led.tcl 2>&1 \
    | grep -E "BITSTREAM OK|ERROR|CRITICAL WARNING|WNS|TNS" | head -30
