#!/usr/bin/env python3
# ============================================================================
# export_int8.py — MobileViTv2 量化与权重导出（W1 模型侧，队友 B 使用）
#
# 职责：把微调好的 PyTorch fp32 模型，导出为 PL 引擎能吃的 INT8 权重 + 描述符。
# 输出与 docs/descriptor_format.md、src/rtl/*.v 的 requantize 公式严格对齐。
#
# 依赖：torch + 本仓库的模型定义（ml-cvnets 或自写）。需 GPU 跑校准集。
#
# 用法：python scripts/quantize/export_int8.py --ckpt model.pth --out build/weights/
#
# 关键公式（与 RTL 完全一致，勿改）：
#   acc  = Σ (x_u8 - zp_in) * w_int8
#   y    = (acc * M + (1 << (shift-1))) >> shift     # 算术右移
#   out  = clamp(y + zp_out, 0, 255)
# ============================================================================
import argparse
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 1. BN 折叠：把 Conv2d + BatchNorm2d 合并成单个 Conv2d
#    weight' = weight * gamma / sqrt(var + eps)
#    bias'   = (bias - mean) * gamma / sqrt(var + eps) + beta
# ---------------------------------------------------------------------------
def fold_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    assert conv.bias is None, "fold 前 conv 不能带 bias（MobileViTv2 的 PW/DW 通常无 bias）"
    w = conv.weight.detach()
    gamma, beta, mean, var, eps = (bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.eps)
    scale = gamma / torch.sqrt(var + eps)
    w_fold = w * scale.view(-1, 1, 1, 1)
    b_fold = (0 - mean) * scale + beta
    conv2 = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                      conv.stride, conv.padding, bias=True)
    conv2.weight.data = w_fold
    conv2.bias.data = b_fold
    return conv2

# ---------------------------------------------------------------------------
# 2. per-channel 量化：把 fp32 权重 -> int8 + (scale -> M, shift)
#    M/shift 由 scale 近似得到：scale ≈ M * 2^-shift，M ∈ [2^30, 2^31)
#    这样 32bit 累加器乘 M 再移位，数值精度足够。
# ---------------------------------------------------------------------------
def scale_to_ms(scale: float):
    import math
    if scale == 0:
        return 0, 0
    # M = round(scale * 2^shift)，shift 使 M 落在 [2^30, 2^31)
    shift = 0
    m = abs(scale)
    while m < (1 << 30):
        m *= 2
        shift += 1
    M = int(round(scale * (1 << shift)))   # scale 可为负，但约定 per-channel scale 取正
    return M, shift

def quantize_weight(w_fp32: np.ndarray) -> tuple[np.ndarray, int, int]:
    """w_fp32: (out_c, in_c, kh, kw) -> (w_int8, M, shift)，per-channel（out_c 维）"""
    out_c = w_fp32.shape[0]
    w_q = np.zeros_like(w_fp32, dtype=np.int8)
    M_list, S_list = [], []
    for oc in range(out_c):
        ch = w_fp32[oc]
        s = np.abs(ch).max() / 127.0
        s = max(s, 1e-9)
        M, shift = scale_to_ms(s)
        q = np.clip(np.round(ch / s), -127, 127).astype(np.int8)
        w_q[oc] = q
        M_list.append(M); S_list.append(shift)
    return w_q, M_list[0], S_list[0]   # TODO: per-channel 混排时逐通道存储（见 descriptor v2）

# ---------------------------------------------------------------------------
# 3. 激活零点 zp 与量化参数（用校准集统计激活范围）
#    zp_in = round(128 - real_min / scale) ... 简化：对称量化 zp=128，scale=max/127
# ---------------------------------------------------------------------------
def activation_qparams(fmap_fp32: np.ndarray) -> tuple[int, int, int]:
    a = fmap_fp32
    scale = max(np.abs(a).max() / 127.0, 1e-9)
    M, shift = scale_to_ms(scale)
    return 128, M, shift          # zp_in=128（对称）

# ---------------------------------------------------------------------------
# 4. 写出（与 sim/scripts/gen_golden*.py 的 .hex 格式一致）
# ---------------------------------------------------------------------------
def dump_hex(path, arr_u8):
    with open(path, "w") as f:
        f.write("\n".join(format(int(v) & 0xFF, "02x") for v in arr_u8.reshape(-1)) + "\n")

def dump_int8_hex(path, arr_i8):
    # int8 权重：two's complement，负值写 0x80..0xFF
    u8 = arr_i8.astype(np.int16) & 0xFF
    dump_hex(path, u8.astype(np.uint8))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="build/weights")
    ap.add_argument("--calib", default=None, help="校准集目录（500 张）")
    args = ap.parse_args()

    print("[export_int8] TODO: 加载模型 + 权重，此处为流程骨架")
    print("[export_int8] 输出目标：")
    print("  每层: weights_int8.hex + (zp_in, zp_out, M, shift)")
    print("  描述符: 按 docs/descriptor_format.md 逐层一行")
    print("[export_int8] 校验：与 RTL 的 requantize 公式一致，用")
    print("  sim/scripts/gen_golden.py 同款镜像做 bit-exact 预比对")

if __name__ == "__main__":
    main()
