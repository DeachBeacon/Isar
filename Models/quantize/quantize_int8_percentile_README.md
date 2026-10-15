# MobileViTv2 的 99.9% 静态 INT8 量化

入口：`quantize_int8_percentile.py`。复用同目录 `quantize_ptq.py` 中的模型构建、数据划分、预处理和 ONNX 导出函数，因此两份 Python 文件需要放在一起。

模型来自 `train.ipynb`：MobileViTv2 0.75、ReLU、10 类；严格加载 `bestmodel_v1_only_weights.pt`。不会执行训练，也不会修改原始权重、训练笔记本或已有量化结果。

## 运行

在 `C:\WorkSpace\qsfpga\Models` 的 PowerShell 中执行。指定环境已有所需依赖；已检查 PyTorch 2.9.1、ONNX 1.22.0、ONNX Runtime 1.31.0。脚本使用 ORT 1.31.0 的校准缓存 API。

运行顶部常量选择的单个配置：

```powershell
& 'C:\ProgramData\anaconda3\envs\py312\python.exe' -X utf8 -u .\quantize_int8_percentile.py
```

完整比较四种组合（同样的 500 张校准图片、全部验证图片）：

```powershell
& 'C:\ProgramData\anaconda3\envs\py312\python.exe' -X utf8 -u .\quantize_int8_percentile.py --compare-all
```

先用少量真实图片检查流程：

```powershell
& 'C:\ProgramData\anaconda3\envs\py312\python.exe' -X utf8 -u .\quantize_int8_percentile.py --compare-all --calib-samples 20 --eval-samples 40
```

少量样本结果只能用于排错，不足以决定最终量化方案。每次运行默认创建新的 `int8_percentile\时间戳` 目录。若指定 `--output-dir`，该目录必须为空，避免覆盖已有结果。

## 配置开关

Python 没有 C/C++ 宏，脚本顶部常量承担相同的配置作用，命令行参数可覆盖常量。

```python
PERCENTILE = 99.9
CLIP_MODE = "abs"          # 或 "central"
CLIP_WEIGHTS = False       # 或 True
COMPARE_ALL = False        # True 时忽略上述两个选择，跑四种组合
WEIGHT_PER_CHANNEL = True  # 权重按输出通道计算百分位并量化
CALIB_SAMPLES = 500
EVAL_SAMPLES = 0           # 0 表示全部验证集
```

| 配置 | 激活截断定义 | 权重截断 |
|---|---|---|
| `abs_activation_only` | `T = P99.9(abs(x))`，限制到 `[-T,T]` | 不预截断，按完整权重范围量化 |
| `abs_activation_and_weight` | 同上 | 每个输出通道使用同样的绝对值百分位规则 |
| `central_activation_only` | `[P0.05(x), P99.95(x)]`，保留中间 99.9% | 不预截断，按完整权重范围量化 |
| `central_activation_and_weight` | 同上 | 每个输出通道使用同样的双尾百分位规则 |

例如，选择双尾截断并同时截断权重：

```powershell
& 'C:\ProgramData\anaconda3\envs\py312\python.exe' -X utf8 -u .\quantize_int8_percentile.py --clip-mode central --clip-weights
```

`--no-clip-weights` 选择只截激活。`--no-per-channel` 改成整张量权重百分位/量化。`--percentile 99.99` 可改变保留比例；双尾模式自动变成两端各截 0.005%。

## 截断和量化实现

- **激活**：在校准集上为各量化边界累计直方图，使用 ONNX Runtime Percentile 校准器。`abs` 显式使用 `symmetric=True` 的绝对值直方图和对称 INT8 激活量化；`central` 显式使用 `symmetric=False` 的有符号值直方图和非对称 INT8 激活量化。默认 8192 个初始直方图桶，可用 `--histogram-bins` 调整。
- **激活阈值是近似百分位**，不是保存全部激活后排序得到的精确百分位。直方图累计过程中桶数可能随范围扩展而增加。推理时由 QuantizeLinear 的饱和操作实现截断，不额外插入浮点 Clip 算子。
- **权重**：先导出并进行 ONNX 预处理，完成 BN 折叠；然后对 Conv/Gemm/MatMul 的常量权重用 `numpy.percentile`（线性插值）计算阈值并执行 `np.clip`，再校准截断权重后的模型。不会截断 bias、BN/LN 的缩放参数或形状常量。
- 权重固定为对称、有符号 INT8；即使双尾截断得到的权重区间不对称，量化 scale 仍按截断后最大绝对值计算。权重数值本身已按双尾阈值截断。默认按输出通道量化：Conv 为第 0 轴；Gemm 按 transB 选择；MatMul 为最后一轴。
- ORT 会结合图结构调整部分共享量化边界，整数网格舍入也会影响有效上下界。因此校准范围与实际饱和边界可能稍有差异。`activation_ranges.json` 保存校准范围，`quantization_parameters.json` 保存图中激活 Q 节点的实际 scale、zero_point 和有效范围。
- 卷积和分类/注意力中的投影采用 W8A8 静态 PTQ，bias 通常为 INT32；LayerNorm、Softmax 等其他算子可保持浮点。输出是 ONNX QDQ INT8 模型，不代表整个网络都用整数算子，也不是 FPGA 的最终编译产物。

ONNX Runtime 的静态量化及校准说明：[官方文档](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)。两种 Percentile 定义对应校准器的 symmetric 开关：[官方实现](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/calibrate.py)。

## 数据和精度比较

默认沿用训练代码的数据路径 `C:\transformerFPGA\MobileViT\Dataset\Driver\train_p`，按 `seed=42`、`val_split=0.2` 做图片级分层划分：训练部分抽取校准样本，验证部分做精度评估。不会对校准图片使用 MixUp/CutMix。

预处理与训练一致：RGB、ToTensor、ImageNet mean/std 标准化，不 Resize。当前数据为 `3×256×256`，所有图片尺寸必须一致。ONNX 仅 batch 维动态。

数据内容、排序或配置变化会影响是否复现原始划分。纯权重没有类别顺序元数据，须保持训练时的 ImageFolder 类别顺序。图片级划分不保证驾驶员隔离。

指定已有独立校准集和评估集时，两者都必须采用相同类别映射的 ImageFolder：

```powershell
& 'C:\ProgramData\anaconda3\envs\py312\python.exe' -X utf8 -u .\quantize_int8_percentile.py --compare-all --data-dir '你的校准集目录' --eval-dir '你的独立评估集目录'
```

FP32 基线始终来自**未经权重截断的原始最佳模型**；所有 INT8 组合使用同一份校准/评估图片及顺序。精度下降按 `本次 FP32 准确率 − 本次 INT8 准确率` 计算，单位为百分点。比较四种配置后，最佳方案只代表本次评估集合中的结果；最终泛化精度需用未参与方案选择的独立测试集确认。

## 输出文件

根目录：

- `model_fp32.onnx`、`model_fp32_preprocessed.onnx`：原始基线及量化前预处理模型。
- `run_config.json`：参数、类别映射、输入尺寸、版本、源权重 SHA256。
- `calibration_samples.json`、`evaluation_samples.json`：实际使用的图片路径和标签。
- `comparison.csv`：四种配置的准确率、下降百分点、预测一致率、大小、耗时及模型路径。
- `best_configuration.json`：本次已运行配置中准确率最高的模型路径；并列时保留先运行者。
- `nodes.json`：预处理图的节点名和算子类型。

每个配置子目录：

- `model_int8_qdq.onnx`：可用 ONNX Runtime CPU 加载的模型。
- `report.json`：准确率、混淆矩阵、逐类准确率等。
- `calibration_cache.json`、`activation_ranges.json`：实际校准缓存及可读范围。
- `quantization_parameters.json`：实际激活量化参数及饱和范围。
- 开启权重截断时另有 `model_weight_clipped_fp32.onnx` 和 `weight_ranges.json`（逐通道阈值及实际截断比例）。

模型输入仍为预处理后的 float32 NCHW 图片，输出为 float32 logits；输入输出类型保持浮点不影响内部 W8A8 量化。推理示例：

```python
import numpy as np
import onnxruntime as ort
from PIL import Image

session = ort.InferenceSession("实际配置目录/model_int8_qdq.onnx",
                               providers=["CPUExecutionProvider"])
with Image.open("一张256x256图片.jpg") as image:
    x = np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1)
x /= np.float32(255.0)
x = (x - np.array([.485, .456, .406], dtype=np.float32)[:, None, None]) / np.array(
    [.229, .224, .225], dtype=np.float32)[:, None, None]
logits = session.run(None, {"images": x[None]})[0]
print("类别下标:", int(logits.argmax(axis=1)[0]))
```
