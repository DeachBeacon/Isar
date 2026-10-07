# MobileViTv2 on ZU19EG — 执行计划（工作稿，非提交物）

> 目标赛事：嵌赛 2026 FPGA 赛道 · 自主选题 · 初级组
> 选题：车载分心驾驶检测（State Farm Distracted Driver Detection，10 类）
> 提交截止：2026-11-03（开发窗口 10/04–10/26，文档与 PPT 10/27–11/02）
> 本文件为团队内部工作稿；仓库提交物须使用纯英文小写路径。

---

## 0. 技术目标与验收指标

| 项 | 目标 |
|---|---|
| 模型 | MobileViTv2-0.75x（XS），INT8 全量化 |
| 输入 | 256×256×3 RGB |
| 类别 | 10 类（c0–c9） |
| 权重位置 | 全部常驻 PL 片内 BRAM（≈2.3MB / 4.3MB） |
| 加速器规模 | **256 个 INT8 MAC @200MHz**（占 1968 DSP 的 13%）；200MHz 下峰值 51.2 GMAC/s，按 60% 效率约 30 GMAC/s，刚好支撑 30fps |
| 帧率目标 | ≥30 fps（256×256） |
| 精度 | 与 PC 0.75x 的 Top-1 掉点 ≤1%（**以量化前的同一模型为基线**计算掉点） |
| 时序 | **200MHz 收敛**（出厂 PS 配置里 PL0 = 200MHz 现成，直接可用）；如需更高频率再由 MMCM 倍频，不作为一期目标 |
| 演示输入 | 暂定 SD 卡读图（采购与摄像头演示延后，先保证部署成功） |
| 基线 | PC PyTorch 0.75x（精度 + 延迟）；可选 A53 裸机 |
| PS 侧 | Vitis standalone 裸机，SD 卡启动，UART/DP 输出 |

**不做**（明确排除，避免工期失控）：MIPI 摄像头输入、PetaLinux 自建、DPU 集成、检测/分割头、1.0x 规模。

---

## 1. 仓库骨架（提交用，纯英文小写）

```
mobilevitv2-z19/
├── readme.md                  # 项目简介 + 复现步骤（含数据获取说明）
├── src/
│   ├── rtl/                   # 自研 RTL：卷积引擎、重排、attention、layernorm
│   ├── ip/                    # 官方 IP 的配置脚本（dma/interconnect/clocking）
│   └── ps/                    # PS 侧裸机 C：调度、SD 读图、UART、基线
├── sim/                       # testbench 与仿真结果
├── build/                     # 可复现构建脚本 + 综合/实现报告
├── board/                     # 上板工程、运行脚本、实测输出
├── data/                      # 测试图与参考结果（不含原数据集，见 readme）
├── skill/                     # 技能包（见第 4 节）
│   └── readme.md
└── report/                    # 设计报告 + ai_log/（大模型协作记录）
```

---

## 2. 逐日排期

### W1 · 端到端最小闭环（10/04–10/10）

| 日期 | 任务 | 检查点 |
|---|---|---|
| 10/04 | 解压 `factory_vivado.zip`，用 `board_test.xpr` / `top.xsa` 的 PS 配置（DDR4、MIO、时钟）与 8 个 `.xdc` 约束作为自己工程的起点（见附录 A） | 能打开并导出自己的 PS 配置 |
| 10/05 | 建自己的 Vivado 工程（CIPS + DDR4 + 时钟 + 复位）；JTAG 下载点灯；串口打印 | 板上点灯 + 串口有输出 |
| 10/06 | SD 卡读图通路：PS 从 SD 读一张 256×256 图 → RAM；UART 回报校验和 | 数据能进 PS |
| 10/07 | 训练侧：0.75x 权重微调（10 类）跑通；导出 fp32 模型 | 微调 Top-1 记录在案 |
| 10/08 | 量化脚本：BN 折叠 + per-channel INT8（PTQ，校准集 500 张）；导出权重 bin | INT8 精度掉点 ≤1% |
| 10/09 | 逐层黄金参考导出：每层输入/输出存 .npy；写比对脚本 | 比对脚本在 PC 上自测通过 |
| 10/10 | 联调：PS 读图 → AXI-Stream 送 PL → 回写 → UART 打印 | 端到端通路打通 |

### W2 · 卷积引擎（10/11–10/17）

| 日期 | 任务 | 检查点 |
|---|---|---|
| 10/11 | 描述符表格式定义（层号/通道/尺寸/stride/权重基址/量化参数） | 文档化，PC 侧脚本能生成 |
| 10/12–10/13 | 1×1 PW 卷积引擎 + line buffer + BRAM 权重读 | 单层仿真比对通过 |
| 10/14–10/15 | 3×3 DW 卷积（stride 1/2）+ padding | 单层仿真比对通过 |
| 10/16 | 第一层（conv_1, 3→24, stride2）上板跑通并与黄金参考比对 | 板上首层 bit-exact |
| 10/17 | 卷积段（L1/L2 的 MV2 块）逐层上板比对；启动权重加载通路 | conv 段全通过 |

### W3 · MobileViT 块与全网络（10/18–10/24）

| 日期 | 任务 | 检查点 |
|---|---|---|
| 10/18 | pixel_unshuffle/pixel_shuffle 地址重排（纯地址逻辑） | 与 PC 重排结果一致 |
| 10/19–10/20 | Q/K/V 1×1 卷积 + P=4 内 softmax（16bit） | attention 前半比对通过 |
| 10/21 | context 计算 + 逐元素乘 + 输出投影 | attention 整块通过 |
| 10/22 | LayerNorm2d（按像素跨通道归约，16bit）+ 残差 + FFN | 单个 MobileViT 块通过 |
| 10/23 | 三层 MobileViT 块串联；分类头（GlobalPool + FC 10） | 全网络板上输出 logits |
| 10/24 | 精度对齐：跑测试集，统计 Top-1；定位偏差层 | 掉点 ≤1%，否则记录原因 |

### W4 · 性能、优化与收尾（10/25–10/26）

| 日期 | 任务 | 检查点 |
|---|---|---|
| 10/25 | 实测 fps/延迟；综合实现报告（LUT/DSP/BRAM/频率）；tiling 与权重常驻策略对比 | 优化前后对比表成稿 |
| 10/26 | 冻结代码；跑 PC 基线（PyTorch/ONNX）与板上数据对齐；整理仓库、构建脚本 | 仓库可被别人从零跑通 |

### 文档与提交（10/27–11/03）

| 日期 | 任务 |
|---|---|
| 10/27–10/28 | 设计报告：选题背景、软硬件划分、接口设计、优化对比表 |
| 10/29 | 技能包 `skill/`：整理 5 个技能项，逐项写适用场景/使用方法/失效条件 |
| 10/30 | 大模型协作记录：提示词、回答、自我纠错轨迹归档 |
| 10/31 | PPT |
| 11/01–11/02 | 复现验证：换一台机器按 readme 走一遍；录制演示视频 |
| 11/03 | 提交 |

---

## 3. 分工（3 人）

| 角色 | 职责 |
|---|---|
| A（PL 计算） | 卷积引擎、行缓存、时序收敛、BRAM 权重映射 |
| B（PL 算子 + 模型） | 重排、attention、LayerNorm、量化与黄金参考、精度对齐 |
| C（系统 + 交付） | Vivado 平台、PS 裸机、AXI/DMA 接口、基线、报告/技能包/仓库 |

跨人约定：描述符表格式由三人共同冻结（10/11 前），此后双方按同一份格式各自开发。

---

## 4. 技能包清单（`skill/`，对应评分项 15 分）

1. **量化与权重导出流程**：PyTorch → BN 折叠 → per-channel INT8 → 权重 bin + 量化参数表。
2. **逐层黄金参考与比对脚本**：自动定位"第一个偏差层"，把硬件调试从猜测变成二分定位。
3. **资源与时序数据自动采集**：解析 Vivado 报告生成 markdown 对比表。
4. **AXI 寄存器映射与 PS 侧调用模板**：`register_map` 生成 C 头文件，避免手写地址。
5. **踩坑清单**：BN 折叠顺序、padding 对齐、pixel_shuffle 地址错位、SD 启动失败排查、DMA 缓存一致性。

每项须写明：适用场景 / 使用方法 / 失效条件 / 从哪些失败中总结。

---

## 5. 风险与降级预案

| 风险 | 触发条件 | 降级动作 |
|---|---|---|
| attention 段未通 | 10/22 仍未比对通过 | PL 只做卷积主干，attention 回 PS 计算，报告写清划分依据 |
| 全片内权重吃紧 | BRAM 布局 >90% | 减少 tiling 缓冲，或把最后 1–2 层权重大小降级（记录在优化对比表） |
| 精度掉点 >1% | 10/24 统计超阈值 | 回退 QAT（用微调环境补几轮），LayerNorm/softmax 提回 16bit |
| 平台起不来 | 10/06 前 SD/JTAG 启动失败 | 用 demo 的 `19EG-image.bin` 启动确认硬件，再回退纯 JTAG 迭代 |
| 工期挤压 | 10/26 仍未全通 | 冻结功能，把已有结果写成"阶段性成果 + 后续计划"如实呈现 |

---

## 6. 大模型协作留痕规范（评分项 15 分）

- 每次有产出的对话，保存提示词、模型回答、以及自我纠错过程，按日期存入 `report/ai_log/`。
- 记录"AI 给了错的东西、我们怎么发现并纠正"的轨迹——这是评分明确要的内容。
- 每提炼出一个可复用技能，同步写入 `skill/` 并注明验证方式。

---

## 附录 A · 本地已有资料清单（D:\BaiduNetdiskDownload\Z19\）

| 位置 | 大小 | 内容与用途 |
|---|---|---|
| `factory_file\factory_vivado.zip` | 91.8MB | ★★★ **出厂 Vivado 工程**：`board_test.xpr`、`top.xsa`、`design_1.bd`、8 个约束（ddr4/eth/mipi/uart/gpio/rs485/extender/system）。**PS 配置与引脚约束直接照抄这个** |
| `course_s1.zip` | 1.6GB | ALINX 基础例程 9541 条目：`01_led`、`02_pl_ddr4_test`（1g64b/2g64b 两套 SODIMM 配置）等 |
| `course_s5_hls.zip` | 1.24GB | HLS 例程 + **MIPI CSI-2 → VDMA → 显示的完整 Vivado 工程**（`03_resizeTry`，含 `.bd`/`.xsa`/`mipi.xdc`）+ rgb2gray/resize/edge 等 HLS IP |
| `lwip例程专用库文件.zip` | 1.8MB | `lwip211` 库（裸机以太网），配 ALINX lwIP 例程使用 |
| `factory_file\petalinux.tar.gz` | 2.25GB | 出厂 PetaLinux 工程（备用，不碰） |
| `factory_file\sd卡出厂镜像.rar` | 4.69GB | 出厂 Linux 演示镜像（备用） |
| `00_hardware.rar` | 130MB | 硬件资料（原理图等），需 RAR 工具 |
| `AXU19EG_AI\` | — | YOLOv3/v5 的 DPU demo：**只有训练环境与 Vitis-AI 文档，没有 Z19 硬件工程**；DPU bitstream 只存在于 SD 镜像的 BOOT.BIN 里 |
| `AXU19EG_AI\SD-card\19EG-image.bin` | 14.88GB | 出厂 SD 整卡镜像（见附录 B） |

## 附录 B · `19EG-image.bin` 结构（已实测）

- **前 512 字节是 PassMark imageUSB 的文件头**（以 UTF-16 的 `imageUSB` 开头），真正的磁盘镜像从偏移 512 开始。
- 偏移 512 起为标准 MBR：分区 1 = FAT32（1MB 起，753MB，启动分区，含 BOOT.BIN / image.ub / system.dtb）；分区 2 = Linux rootfs（754MB 起，4790MB）。其余约 9GB 未分区。
- 镜像内**没有独立 `.bit` 文件**，DPU bitstream 已烧进 BOOT.BIN，无法单独替换 PL。
- 写卡方式：用包内 `install_package\imageUSB.exe`；若用 balenaEtcher/Win32DiskImager，需先剥掉前 512 字节，否则大概率不启动。
- 用途定位：**仅作硬件对照**（确认板子/串口/显示正常）。主线（裸机 + JTAG）不需要它。

## 附录 C · 出厂工程的 PS 实测配置（读自 `top.xsa`）

**器件**：`xczu19eg-ffvc1760-2-i`（与手册一致）；**工程工具版本：Vivado 2020.1**——用 2025.2 直接升级这个 5 年前的工程有很大风险，**建议在 2025.2 里新建工程，照抄参数，不要升级旧工程**。

| 项 | 实测值 |
|---|---|
| PS DDR4 | DDR4、64bit、4 颗 ×16bit 器件、容量 16384 Mbit/颗（共 8GB） |
| DDR 速率 | `SPEED_BIN=DDR4_2400P`，DDR 时钟 1200MHz（2400 MT/s），CL=16 / CWL=16 |
| ECC | Disabled |
| DDR 地址窗口 | 低 0x8_0000_0000 起，高 0x1_FFFF_FFFF（8GB，36 位寻址） |
| PL0 时钟 | **200MHz，已使能**（`FPGA_PL0_ENABLE=1`，源 RPLL）；PL1/PL2/PL3 关闭 |
| A53 (APU) | 请求 600MHz，实际 **525MHz**（做 A53 基线时按 525MHz 估算） |
| PS 主口 | `M_AXI_HPM0_LPD`（32bit）→ PL；PL→PS 用 **S_AXI_HP0_FPD**（BD 里挂 `saxihp0_fpd_aclk`），HP0_DDR_LOW 映射 0x0–0x7FFF_FFFF |
| 已启用外设 | SD0/SD1、QSPI、UART0/UART1、ENET0、ENET3、USB0、DisplayPort+DPAUX、PCIe、I2C0/I2C1、CAN0/CAN1、GPIO(EMIO+MIO)、TTC0–3、WDT |
| IP 地址示例 | PL 侧 IP 映射在 0x8002_0000 / 0x8006_0000 / 0x800B_0000 等（M_AXI_HPM0_LPD 视图） |

**可直接复用的文件**（已提取到 `refs/factory_vivado/`）：`top.bit`（出厂比特流，可先用它 JTAG 验证板子）、`psu_init.c`（PS 初始化代码）、`design_1.hwh`（全量参数）、8 个 `.xdc` 约束。

**硬件资料**（已从 `00_hardware.rar` 提取到 `refs/hardware/`）：`Z19原理图.pdf`、`Z19管脚定义.xls`、`FMC_Length-Z19.xls`、`Z19尺寸结构.pdf/dxf`，以及芯片手册（ug1085 TRM、ds925、ug576 GTH、DDR4/eMMC/GPHY/QSFP28 等）。

## 附录 D · 第三方 IP 使用策略（对应"文档与可复现性"评分）

**原则：接口与搬运用官方 IP，核心算子自研 RTL，PS 侧工具链用开源生态。**

| 层 | 采用 | 理由 |
|---|---|---|
| 总线/搬运/时钟/复位（AXI DMA、SmartConnect、AXI-Stream FIFO、Clocking Wizard、BRAM Controller、Reset） | AMD 官方 IP | 免费、工具集成、时序有保证，赛题明确允许 |
| 基础算术（流水乘法器、DSP48 宏、累加器） | Vivado IP Catalog | 比任何第三方实现可靠 |
| **DW/PW 卷积引擎、pixel_unshuffle/shuffle 重排、可分离自注意力、LayerNorm2d** | **自研 RTL（必须）** | 创新性 30 分 + 初级组基本功考察的核心 |
| PS 侧脚本（量化、逐层比对、黄金参考、报告生成） | 开源 Python 生态（PyTorch/ONNX/numpy/pandas） | 省时间，不影响 FPGA 部分原创性 |
| 完整加速器框架（NVDLA / VTA / FINN / hls4ml） | **不使用** | SoC 级集成成本远超收益，作品会变成"别人的系统" |
| 第三方 HDL 卷积核 | 原则上不用；若坚持，**时间盒 1 天**验证（能否综合、时序能否过、能否接上描述符表），不通立即放弃 | 改造成本常接近重写 |

**合规要求**：引入任何第三方代码前先看 LICENSE，必须允许再分发且与 MIT/Apache-2.0 兼容（GPL/LGPL 会污染整包许可）；在 `readme.md` 与设计报告中维护"第三方组件清单"（名称、URL、commit、许可证）。



