# 逐层黄金参考与比对流程

> 评分对齐：本项对应"校验脚本：自动比对黄金参考"，验收标准是**可复用性**——
> 一支陌生队伍拿到本目录，能直接用在自己的题目上。

## 1. 适用场景

- 自研 RTL 加速器与 PC 参考模型（PyTorch / numpy）逐层对不上的时候
- 不确定"错误从哪一层开始"，靠肉眼波形无法定位
- 需要在换模型/换板卡后快速重建可信的验证基线

## 2. 核心思路：把"猜 bug"变成"二分定位"

```
PC 参考模型 ──导出──> 每层输入/输出 .npy ──转 hex──> RTL testbench
                                                       ↓
                                            逐层比对，输出"第一个不一致的层"
```

只要每层的输入输出都能对齐，硬件调试就退化为**找到第一个偏差层**这一件事，
上游全对、下游全错，问题必然在该层内。

## 3. 使用方法（本项目已落地）

一个自包含的最小实例就是 `pw_conv_engine` 这一层的验证：

```bash
# 1. 生成随机配置 + 激励 + 黄金输出（bit-exact 的软件镜像）
cd sim && python scripts/gen_golden.py 20261014
#    -> data/{input,weights,golden,params}.hex + params.txt

# 2. 跑仿真，testbench 自动比对并打印结论
cd .. && bash build/run_sim.sh

# 3. 多随机种子回归（6 组，防止只对某一组输入成立）
bash build/run_regression.sh
```

**关键设计：配置走文件不走命令行。**
`params.hex` 存 `CIN/COUT/H/W/ZP_IN/ZP_OUT/MUL/SHIFT`，
testbench 用 `$readmemh` 读取 → 同参数集 + 同数据 = 可复现结果，
且跨平台（规避 Git Bash 传 `=` 的问题，见 `skill/pitfalls.md` 第 3 条）。

## 4. 扩展到真实网络（PyTorch 侧）

```python
# 思路骨架：注册 forward hook，把每层输入输出转成硬件能吃的格式
ref = {}
def hook(name):
    def fn(module, inp, out):
        ref[name] = (inp[0].detach().numpy(), out.detach().numpy())
    return fn

for name, m in model.named_modules():
    if name in target_layers:
        m.register_forward_hook(hook(name))

# 量化后导出：uint8(激活) / int8(权重) / (zp, mul, shift)
# 硬件要求：激活 uint8 且 hw 顺序为 CHW，拍平后逐字节写入 .hex
```

对每一层执行：
1. 把**上一层 RTL 的实际输出**当作本层输入（避免误差累积掩盖问题）
2. 与 PyTorch 该层的输出比对，统计 `max_abs_diff` 与不一致元素占比
3. 报告第一个不一致层 + 差异统计

## 5. 失效条件（什么时候这套办法不好用）

- **浮点层**：attention 的 softmax、LayerNorm 若先用 float 实现，
  没有唯一"正确答案"，需要先定义容差（本项目约定 16bit 定点 + 容差阈值）
- **有状态层**：涉及跨帧缓存的模块，不能按单层独立比对
- **量化误差累积**：若刻意用"整网端到端精度"对比，单层 bit-exact 就不再是目标，
  此时应改用 Top-1 掉点作为验收（见 plan.md 的 ≤1% 指标）

## 6. 已验证的效果

| 日期 | 验证对象 | 结果 |
|---|---|---|
| 2026-10-14 | `pw_conv_engine` 1×1 PW 卷积层 | 6 组随机种子全 PASS，300 输出 bit-exact |
| 2026-10-14 | 该流程定位 BRAM 写口 x 值故障 | 20 分钟内定位并修复（见 pitfalls.md 第 1 条） |

## 7. 复用清单（换题目/换板卡要改什么）

| 组件 | 是否需要改 | 说明 |
|---|---|---|
| `gen_golden.py` 的算术函数 | 改 | 换成新算子的量化公式 |
| `params.hex` 字段 | 改 | 换成新算子的配置字段 |
| `tb_*.v` 的搬运骨架 | 基本不改 | valid/ready 送入、收集、比对 |
| `run_sim.sh` / `run_regression.sh` | 不改 | 工具路径按机器调整 |
