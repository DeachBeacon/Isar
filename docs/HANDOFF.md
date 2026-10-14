# 使用说明 / 交付交接书（HANDOFF）

> 给上板队友的**总入口**。按顺序做即可，每一步都有验收标准。

## 0. 三句话搞清楚现状

1. **仿真侧（W2 核心）已全部完成并验证**：PW/DW 卷积引擎 6 组随机种子 bit-exact 通过，跑 `bash build/run_regression.sh` 应看到 `12 passed, 0 failed`。
2. **板卡侧（W1 平台）还没动**：因为板子不在手上，我把所有材料准备好了，你拿到板子按本文档做即可。
3. **分工**：RTL + 仿真我来继续；**你负责平台打通 + 上板**；队友 B 负责训练/量化。

## 1. 仿真复现（先验证你机器环境 OK，5 分钟）

```bash
cd <仓库根>
# 确认 Vivado 路径（不在 D:/vivado 就 export VIVADO_BIN=你的路径/bin）
export VIVADO_BIN="D:/vivado/2025.2/Vivado/bin"

bash build/run_regression.sh
# 期望：pw/dw 各 6 组，合计 regression: 12 passed, 0 failed
```

## 2. 上板冒烟（拿到板子后第一步，10 分钟）

1. 板子供电 + USB-JTAG 连电脑
2. Vivado GUI → Hardware Manager → Auto Connect → Program Device
3. 选 `board/led_blink/led_blink.bit`
4. **板上 LED 闪烁 = 成功**（板子/时钟/JTAG 全 OK）

## 3. 平台打通（核心工作，参考 docs/platform_bringup.md）

按顺序：点灯 → 串口 → SD 读图 → DMA 环回。每个环节参照 `E:\Z19\extracted\course_s1\` 对应例程。
关键 PS 参数照抄 `plan.md` 附录 C。

## 4. 引擎上板（W2 收尾）

- DMA 的 AXI-Stream 接引擎 `in_*`/`out_*`
- 权重走 AXI-Lite 写 `wt_*` 口，写完拉 `start`
- 与 `sim/scripts/gen_golden*.py` 的黄金参考比对（板上首层 bit-exact 是 10/16 检查点）

## 5. 三条铁律（踩坑总结，详见 skill/pitfalls.md）

1. **旧 2020.1 工程不要升级**，照抄参数新建 2025.2 工程
2. 路径含空格时 `read_verilog` 用相对路径；Git Bash 调 .bat 不传 `=`
3. 仿真配置走文件（`sim/data/params.hex`），不走命令行参数
