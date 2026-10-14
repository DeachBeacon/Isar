#!/usr/bin/env python3
# ============================================================================
# gen_golden.py — generate randomized layer config + stimulus + golden output
# for pw_conv_engine simulation. Bit-exact software mirror of the RTL.
#
# Usage:  python gen_golden.py [seed]     (run from sim/ directory)
# Output: data/input.hex, weights.hex, golden.hex, params.hex, params.txt
# The requantization here MUST stay in sync with pw_conv_engine.v:
#   acc  = sum_ci (in_u8 - zp_in) * w_int8
#   y    = (acc * M + (1 << (shift-1))) >> shift     (arithmetic shift)
#   out  = clamp(y + zp_out, 0, 255)
# ============================================================================
import random
import sys

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 20261014
rng = random.Random(seed)

# ---- layer configuration (small enough for a quick sim) ----
CIN   = rng.randint(3, 8)
COUT  = rng.randint(4, 16)
H     = rng.randint(4, 8)
W     = rng.randint(4, 8)
ZP_IN  = rng.randint(110, 130)
ZP_OUT = rng.randint(110, 130)
M     = rng.randint(1 << 19, 1 << 20)      # multiplier, positive
SHIFT = rng.randint(14, 18)                # >= 1 required by RTL

HW = H * W
IN_N  = CIN * HW
OUT_N = COUT * HW
WT_N  = CIN * COUT

# ---- stimulus ----
inp = [rng.randint(max(0, ZP_IN - 40), min(255, ZP_IN + 40)) for _ in range(IN_N)]
wt  = [rng.randint(-128, 127) for _ in range(WT_N)]

# ---- golden: exact mirror of RTL arithmetic ----
def requant(acc):
    prod = acc * M
    rounded = prod + (1 << (SHIFT - 1))
    y = rounded >> SHIFT          # python >> is arithmetic (floor) for ints
    val = y + ZP_OUT
    return 0 if val < 0 else (255 if val > 255 else val)

golden = []
for oc in range(COUT):
    for p in range(HW):
        acc = 0
        for ci in range(CIN):
            x = inp[ci * HW + p] - ZP_IN
            acc += x * wt[oc * CIN + ci]
        golden.append(requant(acc))

# ---- write files (one byte per line, 2 hex digits) ----
def dump(path, data):
    with open(path, "w") as f:
        f.write("\n".join(format(v & 0xFF, "02x") for v in data) + "\n")

dump("data/input.hex", inp)
dump("data/weights.hex", wt)
dump("data/golden.hex", golden)

# params.hex: one 32-bit hex word per line, read by testbench $readmemh
with open("data/params.hex", "w") as f:
    for v in (CIN, COUT, H, W, ZP_IN, ZP_OUT, M, SHIFT):
        f.write(format(v, "08x") + "\n")

with open("data/params.txt", "w") as f:
    f.write(f"CIN={CIN}\nCOUT={COUT}\nH={H}\nW={W}\n"
            f"ZP_IN={ZP_IN}\nZP_OUT={ZP_OUT}\nM={M}\nSHIFT={SHIFT}\n")

print(f"seed={seed} CIN={CIN} COUT={COUT} H={H} W={W} "
      f"ZP_IN={ZP_IN} ZP_OUT={ZP_OUT} M={M} SHIFT={SHIFT}")
print(f"in={IN_N}B wt={WT_N}B out={OUT_N}B")
