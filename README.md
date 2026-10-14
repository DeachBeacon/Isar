# Isar

10TH ANNIVERSARY FPGA INNOVATIVE DESIGN COMPETITION

这里是嵌赛 FPGA 的仓库。

## 项目

车载分心驾驶检测加速器 —— MobileViTv2-0.75x INT8 全量化，自研 RTL 实现，
目标器件 XCZU19EG-FFVC1760-2-I（ALINX ZU19EG / AXU19EG 板卡）。

- 选题：State Farm Distracted Driver Detection（10 类）
- 目标：256×256 输入，≥30 fps，精度掉点 ≤1%
- 方案：PL 侧自研 DW/PW 卷积 + attention + LayerNorm；PS 侧 Vitis 裸机调度
- 详细计划见 [plan.md](plan.md)

苹果模型仓库地址：https://github.com/apple/ml-cvnets

## 目录结构

```
├── src/
│   ├── rtl/       自研 RTL（卷积引擎、重排、attention、layernorm）
│   ├── ip/        官方 IP 配置脚本（dma / interconnect / clocking）
│   └── ps/        PS 侧裸机 C
├── sim/           testbench 与仿真脚本
├── build/         可复现构建脚本 + 综合实现报告
├── board/         上板工程、运行脚本、实测输出
├── data/          测试数据与参考结果
├── docs/          接口与格式文档
├── skill/         技能包（大模型协作沉淀）
└── report/        设计报告 + ai_log/（AI 协作记录）
```

## 复现步骤（当前进度：W2 卷积引擎）

### 环境

- Vivado 2025.2（官方推荐 2026.1 / 2025.2）
- Python 3.8+（生成黄金参考数据，无需额外依赖）
- Windows + Git Bash（Vivado 路径可用环境变量 `VIVADO_BIN` 覆盖）

### 运行仿真（PW + DW 卷积引擎）

```bash
export VIVADO_BIN="D:/vivado/2025.2/Vivado/bin"   # 不在 D 盘时改路径

# 多随机种子回归（PW + DW 各 6 组，共 12 项）
bash build/run_regression.sh
# 期望：regression: 12 passed, 0 failed

# 单个模块
bash build/run_sim.sh      # PW，配合 sim/scripts/gen_golden.py
bash build/run_sim_dw.sh   # DW，配合 sim/scripts/gen_golden_dw.py
```

### 生成 LED 点灯 bitstream（上板冒烟测试）

```bash
bash build/build_led.sh
# 产出 board/led_blink/led_blink.bit
```

### 上板 / 交付

- 上板步骤见 [docs/HANDOFF.md](docs/HANDOFF.md)
- 材料清单见 [docs/MATERIALS.md](docs/MATERIALS.md)
- 平台打通见 [docs/platform_bringup.md](docs/platform_bringup.md)

### 数据集

State Farm Distracted Driver Detection 数据集**未包含在仓库中**，请自行获取：
https://www.kaggle.com/c/state-farm-distracted-driver-detection

## 验证状态

| 模块 | 状态 | 验证方式 |
|---|---|---|
| `pw_conv_engine` (1×1 PW, INT8) | 6 种子 bit-exact | `sim/tb/tb_pw_conv_engine.v` |
| `dw_conv_engine` (3×3 DW, stride1/2+pad) | 6 种子 bit-exact | `sim/tb/tb_dw_conv_engine.v` |
| `led_blink` (平台冒烟) | bitstream 已生成，时序收敛 | `build/build_led.sh` |
| 重排 / attention / LayerNorm | 待开发（W3） | — |


## 第三方组件清单

| 组件 | 来源 | 许可证 |
|---|---|---|
| AXI DMA / SmartConnect / Clocking Wizard 等 | Vivado IP Catalog | AMD 免费授权 |
| 暂无外部开源代码引入 | — | — |

## 许可证

MIT / Apache-2.0（待队伍确认）
