# 描述符表格式 v1（内部约定）

> 状态：**草案，待三人冻结**（plan.md W2 约定）
> 用途：PL 侧每一层算子的配置来源；PC 侧脚本按此格式生成；PS 侧按此格式写寄存器
> 变更需三人确认并更新本文件版本号

## 1. 为什么需要描述符表

MobileViTv2 由几十层不同形状的算子串联（PW/DW 卷积、重排、attention、LayerNorm、FC）。
如果每层都做一个独立硬件模块，PL 面积会爆炸、也无法复用调试成果。正确做法是：
**一套可重配的算子引擎 + 一张描述表**，PS 按行下发配置，引擎按行执行。

好处：① 硬件只有几个引擎，改动集中在描述符生成脚本；② 换模型（换题目）只需换表；
③ 调试时能精确复现某一层（黄金参考比对的前提）。

## 2. 表结构

每层占 **1 行**，字段如下（v1）：

| 序号 | 字段 | 位宽 | 说明 |
|---|---|---|---|
| 0 | `layer_id` | 16 | 层序号，从 0 递增；也用于日志与黄金参考文件名 |
| 1 | `op_type` | 8 | 0=PW_CONV, 1=DW_CONV, 2=PIXEL_UNSHUFFLE, 3=ATTENTION, 4=LAYERNORM, 5=FC |
| 2 | `cin` | 16 | 输入通道数 |
| 3 | `cout` | 16 | 输出通道数 |
| 4 | `h_in` | 16 | 输入特征图高 |
| 5 | `w_in` | 16 | 输入特征图宽 |
| 6 | `kh` | 8 | 卷积核高（PW=1，DW=3） |
| 7 | `kw` | 8 | 卷积核宽 |
| 8 | `stride_h` | 8 | 行方向步长 |
| 9 | `stride_w` | 8 | 列方向步长 |
| 10 | `pad_h` | 8 | 行方向 padding（DW 用） |
| 11 | `pad_w` | 8 | 列方向 padding |
| 12 | `wt_base` | 32 | 该层权重在权重 BRAM 中的**字节基址** |
| 13 | `bias_base` | 32 | 偏置基址（预留，当前为 0） |
| 14 | `zp_in` | 8 | 输入零点（uint8 域） |
| 15 | `zp_out` | 8 | 输出零点（uint8 域） |
| 16 | `mul` | 32 | 量化乘数 M（int32，恒正） |
| 17 | `shift` | 8 | 量化右移位数（≥1） |
| 18 | `act` | 8 | 激活类型：0=ReLU6, 1=SiLU, 2=无, 3=softmax（层内） |
| 19 | `flags` | 8 | bit0=接残差, bit1=需 16bit 精度, bit2=最后层 |
| 20 | `reserved` | 32 | 对齐预留 |

**行宽 = 21 个 32bit 字 = 84 字节**（v1，未压缩；后续可按需压到 16 字以内）。

## 3. 量化参数的产生（与 RTL 的一致性要求）

RTL 的 requantize 公式（`pw_conv_engine.v`）：

```
acc = Σ_ci (x_u8 - zp_in) * w_int8
y   = (acc * mul + (1 << (shift-1))) >> shift      // 算术右移
out = clamp(y + zp_out, 0, 255)
```

PC 侧量化脚本必须产出**同一公式**的 `(zp_in, zp_out, mul, shift)` 四元组，
否则硬件与软件模型必然对不上。per-channel 量化时，若不同输出通道 scale 不同，
v1 约定：**引擎一次只处理一个 scale 组**（描述符按 scale 组拆行）。
后续如需 per-channel 单层内混合，另开 v2 讨论（会增加 mul/shift 存储量）。

## 4. 与 RTL 的接口对应

`pw_conv_engine.v` 当前消费的字段：`cin, cout, h_in(=H), w_in(=W), zp_in, zp_out, mul, shift`
（PW 卷积 kh=kw=1，stride=1，pad=0，用不到其他字段）。

DW 引擎将消费 `kh, kw, stride_*, pad_*`。

## 5. 变更记录

| 版本 | 日期 | 变更 | 确认人 |
|---|---|---|---|
| v1 | 2026-10-14 | 初稿，21 字段 | 待 A/B/C 确认 |
