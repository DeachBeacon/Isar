# 材料清单（MATERIALS）

> 打包交给上板队友前，对照本清单核对。标注「✅已产出 / ⬜需队友补齐」。

## 一、软件环境（队友机器上需装）

| 项 | 版本 | 备注 |
|---|---|---|
| Vivado / Vitis | **2025.2** | 已确认可用；安装路径任意，脚本用 `VIVADO_BIN` 环境变量覆盖 |
| Python | 3.8+ | 仅标准库即可跑黄金参考（无需 numpy/torch） |
| Git Bash | 任意 | Windows 下跑脚本用 |
| 串口工具 | 任意 | 看 UART 输出 |

## 二、硬件

| 项 | 状态 |
|---|---|
| ALINX Z19 / AXU19EG 板卡 | ⬜ 队友处 |
| 电源 + JTAG USB 线 | ⬜ 队友处 |
| SD 卡（W4 演示用，W1–W3 不需要） | ⬜ 队友处 |

## 三、仓库内已产出（✅）

| 路径 | 内容 |
|---|---|
| `src/rtl/pw_conv_engine.v` | PW 1×1 卷积引擎（INT8，已验证） |
| `src/rtl/dw_conv_engine.v` | DW 3×3 卷积引擎（stride1/2+pad，已验证） |
| `sim/` | testbench + 黄金参考生成器 + 数据 |
| `build/run_sim.sh` `run_sim_dw.sh` `run_regression.sh` | 仿真/回归脚本 |
| `board/led_blink/led_blink.bit` | **已生成的点灯 bitstream**（8.78MB） |
| `build/build_led.sh` `build_led.tcl` | 点灯工程重建脚本 |
| `build/create_project.tcl` | 一键建 RTL 工程 |
| `src/ps/` | PS 裸机 C 模板（uart/sd/dma） |
| `scripts/quantize/export_int8.py` | 量化导出脚本（模型侧骨架） |
| `docs/` | descriptor_format / platform_bringup / handoff |
| `refs/factory_vivado/` | 出厂工程（PS 配置 + 约束 + ip_repo） |
| `skill/` `report/ai_log/` | 技能包 + AI 协作记录 |
| `plan.md` | 执行计划 v2 |

## 四、仓库外（队友机器上，⬜）

| 项 | 位置 | 用途 |
|---|---|---|
| ALINX 例程（01_led 等 9 个） | `E:\Z19\extracted\course_s1\` | W1 各步骤参照 |
| ALINX 教程 PDF（s0–s5） | `E:\Z19\*.pdf` | 学习参照 |
| 出厂镜像/硬件资料 | `E:\Z19\` | W4 演示 + 原理图 |
| MobileViTv2 微调权重 | 队友 B 产出 | 量化脚本输入 |
| State Farm 数据集 | Kaggle（自行下载） | 训练/校准 |

## 五、打包建议

```bash
# 在仓库根目录
git add -A && git commit -m "W1/W2 deliverables" && git push
# 或直接压缩源码（不含 .bit/.git 也可以让队友 git clone）
tar -czf isar_handoff.tar.gz --exclude=.git --exclude='*.bit' --exclude='*.runs' .
```

最小可行集 = `src/rtl` + `sim` + `build` + `board/led_blink` + `docs` + `refs/factory_vivado`。
