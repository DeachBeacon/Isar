#!/usr/bin/env python3
# ============================================================================
# gen_golden_dw.py — randomized config + stimulus + golden for dw_conv_engine
# Bit-exact software mirror of the RTL (same requantize formula as pw engine).
# Usage: python gen_golden_dw.py [seed]   (run from sim/)
# Output: data/input.hex, weights.hex, golden.hex, params.hex, params.txt
# ============================================================================
import random
import sys

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 20261014
rng = random.Random(seed)

C     = rng.randint(3, 8)        # cin == cout for DW
H     = rng.randint(5, 9)
W     = rng.randint(5, 9)
STRIDE = rng.choice([1, 2])
PAD    = rng.choice([0, 1])
ZP_IN  = rng.randint(110, 130)
ZP_OUT = rng.randint(110, 130)
M     = rng.randint(1 << 19, 1 << 20)
SHIFT = rng.randint(14, 18)

HW   = H * W
IN_N = C * HW
WT_N = C * 9
HOUT = (H + 2 * PAD - 3) // STRIDE + 1
WOUT = (W + 2 * PAD - 3) // STRIDE + 1
OUT_N = C * HOUT * WOUT

inp = [rng.randint(max(0, ZP_IN - 40), min(255, ZP_IN + 40)) for _ in range(IN_N)]
wt  = [rng.randint(-128, 127) for _ in range(WT_N)]

def requant(acc):
    prod = acc * M
    rounded = prod + (1 << (SHIFT - 1))
    y = rounded >> SHIFT
    val = y + ZP_OUT
    return 0 if val < 0 else (255 if val > 255 else val)

golden = []
for oc in range(C):
    for oy in range(HOUT):
        for ox in range(WOUT):
            acc = 0
            for ky in range(3):
                for kx in range(3):
                    iy = oy * STRIDE - PAD + ky
                    ix = ox * STRIDE - PAD + kx
                    if 0 <= iy < H and 0 <= ix < W:
                        acc += (inp[oc * HW + iy * W + ix] - ZP_IN) * wt[oc * 9 + ky * 3 + kx]
            golden.append(requant(acc))

def dump(path, data):
    with open(path, "w") as f:
        f.write("\n".join(format(v & 0xFF, "02x") for v in data) + "\n")

dump("data/input.hex", inp)
dump("data/weights.hex", wt)
dump("data/golden.hex", golden)

# params.hex order: CIN H W STRIDE PAD ZP_IN ZP_OUT MUL SHIFT
with open("data/params.hex", "w") as f:
    for v in (C, H, W, STRIDE, PAD, ZP_IN, ZP_OUT, M, SHIFT):
        f.write(format(v, "08x") + "\n")

with open("data/params.txt", "w") as f:
    f.write(f"C={C}\nH={H}\nW={W}\nSTRIDE={STRIDE}\nPAD={PAD}\n"
            f"ZP_IN={ZP_IN}\nZP_OUT={ZP_OUT}\nM={M}\nSHIFT={SHIFT}\n"
            f"HOUT={HOUT}\nWOUT={WOUT}\n")

print(f"seed={seed} C={C} H={H} W={W} STRIDE={STRIDE} PAD={PAD} "
      f"ZP_IN={ZP_IN} ZP_OUT={ZP_OUT} M={M} SHIFT={SHIFT} "
      f"HOUT={HOUT} WOUT={WOUT}")
print(f"in={IN_N}B wt={WT_N}B out={OUT_N}B")
