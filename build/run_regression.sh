#!/bin/bash
# ============================================================================
# run_regression.sh — multi-seed bit-exact regression for pw_conv_engine
# Usage: bash build/run_regression.sh
# ============================================================================
cd "$(dirname "$0")/.."
PYEXE=$(ls "C:/Users/Justin/.workbuddy/binaries/python/envs/default/python.exe" 2>/dev/null \
     || ls "C:/Users/Justin/.workbuddy/binaries/python/versions/3.13.12/python.exe")

pass=0; fail=0
for s in 1 7 42 1234 20261014 99999; do
    (cd sim && "$PYEXE" scripts/gen_golden.py $s > /dev/null)
    line=$(bash build/run_sim.sh 2>&1 | grep -E "^RESULT")
    echo "seed=$s $line"
    case "$line" in
        *PASS*) pass=$((pass+1)) ;;
        *)      fail=$((fail+1)) ;;
    esac
done
echo "----------------------------------------"
echo "regression: $pass passed, $fail failed"
[ $fail -eq 0 ]
