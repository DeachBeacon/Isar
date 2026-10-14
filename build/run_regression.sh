#!/bin/bash
# ============================================================================
# run_regression.sh — multi-seed bit-exact regression for pw + dw engines
# Usage: bash build/run_regression.sh
# ============================================================================
cd "$(dirname "$0")/.."
PYEXE=$(ls "C:/Users/Justin/.workbuddy/binaries/python/envs/default/python.exe" 2>/dev/null \
     || ls "C:/Users/Justin/.workbuddy/binaries/python/versions/3.13.12/python.exe")

pass=0; fail=0
run_one() {
    local mod="$1" gen="$2" sim="$3" seed="$4"
    (cd sim && "$PYEXE" "$gen" "$seed" > /dev/null)
    local line=$(bash "$sim" 2>&1 | grep -E "^RESULT")
    echo "$mod seed=$seed $line"
    case "$line" in
        *PASS*) pass=$((pass+1)) ;;
        *)      fail=$((fail+1)) ;;
    esac
}

for s in 1 7 42 1234 20261014 99999; do
    run_one "pw" "scripts/gen_golden.py"    "build/run_sim.sh"    "$s"
    run_one "dw" "scripts/gen_golden_dw.py" "build/run_sim_dw.sh" "$s"
done
echo "----------------------------------------"
echo "regression: $pass passed, $fail failed"
[ $fail -eq 0 ]
