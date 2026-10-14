# src/ps — PS 侧裸机 C（Vitis standalone）

> 说明：这些是**模板**，需要先用 Vitis 导出 `.xsa` 平台并建 BSP 才能编译。
> 完整可编译例程见 ALINX `course_s1`（04/07/11/22/23/24）。

| 文件 | 用途 | 依赖 BSP 库 | W1 对应步骤 |
|---|---|---|---|
| `uart_hello.c` | PS 启动 + 串口打印 | standalone | 第二步（串口） |
| `sd_read_image.c` | SD 读一张 256×256×3 图 | xilffs / xilsecure / xilpm | 第三步（读图） |
| `dma_loopback.c` | PL↔PS DMA 环回 | xaxidma | 第五步（DMA 通路） |

## 编译步骤（队友用）

1. Vivado 里打开工程 → File → Export → Export Hardware（含 bitstream，得到 `.xsa`）
2. Vitis 2025.2 → Create Platform Project → 选 `.xsa`
3. Create Application Project → 选该平台 → Empty Application（C）
4. 把本目录对应 `.c` 拷进 `src/`，Board Support Package 里勾选所需库
5. Build → Run As → Launch on Hardware（JTAG）

## 与 W2 引擎对接约定

- 引擎输入输出都是 AXI-Stream；DMA 的 `M_AXIS`/`S_AXIS` 直接接
- 权重写入：用 AXI-Lite 主口写引擎的 `wt_waddr/wt_wdata/wt_we`，写完再拉 `start`
- 描述符字段（cin/cout/h/w/mul/shift/zp...）见 `docs/descriptor_format.md`
- 图像通道顺序：全链路统一 **CHW**（channel 平面优先），量化脚本导出也按 CHW
