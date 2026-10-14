# W1 平台打通指南（上板队友必读）

> 目标：在 Vivado 2025.2 里建起带 PS + DDR4 的工程，完成"点灯 → 串口 → SD 读图 → DMA 通路"。
> 原则（plan.md 附录 C）：**新建工程照抄参数，不要升级 2020.1 旧工程**。

## 0. 前置材料（都在仓库里 / E:\Z19）

| 材料 | 位置 |
|---|---|
| 出厂工程（PS 配置参考） | `refs/factory_vivado/`（含 `design_1.bd`、`top.xsa`、8 个 `.xdc`） |
| 出厂 PS 实测配置 | `plan.md` 附录 C |
| LED 冒烟测试 bitstream | `board/led_blink/led_blink.bit`（已生成） |
| 基础例程 | `E:\Z19\extracted\course_s1\`（01_led / 04_ps_hello / 07_ps_uart / 11_ps_sd / 24_dma_loopback） |
| 用户手册 | `Z19用户手册.pdf`（第 6 章时钟、第 18+ 页引脚表） |

## 1. 第一步：验证板子（10 分钟）

1. 板子接电源、USB-JTAG 线连电脑
2. Vivado GUI → Open Hardware Manager → Open Target → Auto Connect
3. Program Device，选 `board/led_blink/led_blink.bit`
4. **板上 LED 闪烁 = 板子、PL 时钟、JTAG 全正常**。不闪则回到供电/线缆/驱动排查

## 2. 第二步：PS 关键参数（照抄，别手滑）

来自 `plan.md` 附录 C（读自出厂 `top.xsa`，已实测）：

| 项 | 值 |
|---|---|
| 器件 | xczu19eg-ffvc1760-2-i |
| DDR4 | 64bit、4 颗 ×16bit、单颗 16384 Mbit、共 8GB |
| DDR 速率 | DDR4_2400P，1200MHz（2400 MT/s），CL=16 / CWL=16 |
| PL0 时钟 | **200MHz 已使能**（源 RPLL） |
| A53 | 请求 600MHz，实际 525MHz |
| PS→PL 主口 | M_AXI_HPM0_LPD（32bit） |
| PL→PS | S_AXI_HP0_FPD（挂 `saxihp0_fpd_aclk`），HP0_DDR_LOW 0x0–0x7FFF_FFFF |
| 已启用外设 | SD0/SD1、UART0/UART1、ENET0/ENET3、USB0、DP、PCIe、I2C0/1、CAN0/1、GPIO、TTC0–3、WDT |

## 3. 第三步：新建 2025.2 工程

1. `source build/create_project.tcl`（或 GUI Tools → Run Tcl Script）
2. Create Block Design → 加 **Zynq UltraScale+ MPSoC IP**（CIPS）→ 按上面参数配置 DDR4、时钟、外设
   - 参考：打开 `refs/factory_vivado/board_test.srcs/sources_1/bd/design_1/design_1.bd`
     （用 Vivado 打开后**只读参数，不直接升级**）
3. 加 `Clocking Wizard`（如需要额外时钟）+ `Processor System Reset`
4. 生成 Output Products → 生成 HDL Wrapper
5. 约束：把 `refs/factory_vivado/.../constrs_1/new/*.xdc` 里的引脚照搬到自己工程

## 4. 第四步：最小闭环（点灯 → 串口 → SD → DMA）

| 步骤 | 参考例程 | 检查点 |
|---|---|---|
| PS Hello + 串口打印 | `course_s1/04_ps_hello`、`07_ps_uart` | 串口有输出 |
| PL 点灯（EMIO/GPIO） | `course_s1/01_led`、`16_ps_axi_gpio` | 板载 LED 受控 |
| SD 读一张图 | `course_s1/11_ps_sd` | 校验和回报正确 |
| PL↔PS DDR / BRAM | `course_s1/22`、`23` | 数据回读一致 |
| **DMA 环回** | `course_s1/24_dma_loopback` | PS→DMA→PL→DMA→PS 数据一致 |

## 5. 与 W2 引擎对接

- PW/DW 引擎的输入输出都是 AXI-Stream（`in_*` / `out_*`）
- 上板时：`S_AXI_HP0` 的 DMA（AXI-DMA IP）`M_AXIS` → 引擎 `in_*`；引擎 `out_*` → DMA `S_AXIS`
- 权重用 AXI-Lite 通过 `wt_waddr/wt_wdata/wt_we` 写入引擎 BRAM（先写权重再拉 `start`）
- 描述符字段见 `docs/descriptor_format.md`

## 6. 常见坑（详见 skill/pitfalls.md）

- 旧工程（2020.1）直接升级到 2025.2 大概率编译不过，务必照抄参数新建
- Git Bash 里 vivado.bat 传 `=` 会失败；工程内路径含空格时 `read_*` 用相对路径
- SD 卡启动镜像（BOOT.BIN）只在 W4 演示用；W1–W3 全程 JTAG 即可
