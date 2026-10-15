"""MobileViTv2 静态 INT8 PTQ：比较两种 99.9% 截断定义及权重截断开关。

Python: C:\\ProgramData\\anaconda3\\envs\\py312\\python.exe
复用同目录 quantize_ptq.py 中与 train.ipynb 一致的建模、数据划分和预处理。
运行方式见 quantize_int8_percentile_README.md。不会执行训练或修改原权重。
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import inspect
import io
import json
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper
from onnxruntime.quantization import (
    CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static,
)
from onnxruntime.quantization.calibrate import create_calibrator, save_tensors_data
from onnxruntime.quantization.shape_inference import quant_pre_process

from quantize_ptq import ROOT, batches, build_model, export_model, make_data


# ================= 配置开关（Python 中对应“宏”）=================
PERCENTILE = 99.9
CLIP_MODE = "abs"             # "abs": |x| 的 P99.9；"central": 两端各截 0.05%
CLIP_WEIGHTS = False          # False: 只截激活；True: 激活和权重都截断
COMPARE_ALL = False           # True: 一次跑完上述 2×2=4 种组合
WEIGHT_PER_CHANNEL = True     # 沿用已有脚本：权重按输出通道量化/截断
CALIB_SAMPLES = 500           # 从训练部分抽取；全部配置使用相同图片
EVAL_SAMPLES = 0              # 0: 完整验证集；正数: 快速检查子集
HISTOGRAM_BINS = 8192         # 激活百分位由累计直方图近似估计
THREADS = 4
OPS = ["Conv", "Gemm", "MatMul"]


def log(message):
    print(message, flush=True)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=ROOT / "bestmodel_v1_only_weights.pt")
    p.add_argument("--data-dir", type=Path,
                   default=Path("C:/transformerFPGA/MobileViT/Dataset/Driver/train_p"))
    p.add_argument("--eval-dir", type=Path, help="可选的独立评估集 ImageFolder")
    p.add_argument("--output-dir", type=Path, help="新建空目录；默认自动按时间命名")
    p.add_argument("--clip-mode", choices=["abs", "central"], default=CLIP_MODE)
    p.add_argument("--clip-weights", action=argparse.BooleanOptionalAction, default=CLIP_WEIGHTS)
    p.add_argument("--compare-all", action=argparse.BooleanOptionalAction, default=COMPARE_ALL)
    p.add_argument("--percentile", type=float, default=PERCENTILE)
    p.add_argument("--per-channel", action=argparse.BooleanOptionalAction, default=WEIGHT_PER_CHANNEL)
    p.add_argument("--calib-samples", type=int, default=CALIB_SAMPLES)
    p.add_argument("--eval-samples", type=int, default=EVAL_SAMPLES)
    p.add_argument("--batch-size", type=int, default=8, help="仅影响评估；校准固定每批 1 张")
    p.add_argument("--histogram-bins", type=int, default=HISTOGRAM_BINS)
    p.add_argument("--threads", type=int, default=THREADS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-split", type=float, default=0.2)
    args = p.parse_args()
    if args.clip_mode not in {"abs", "central"}:
        p.error('CLIP_MODE 必须是 "abs" 或 "central"。')
    if not 0 < args.percentile <= 100:
        p.error("percentile 必须在 (0,100]；99.9 应写 99.9 而非 0.999。")
    if min(args.calib_samples, args.batch_size, args.threads) <= 0 or args.eval_samples < 0:
        p.error("calib-samples/batch-size/threads 必须为正；eval-samples 必须非负。")
    if not 0 < args.val_split < 1 or args.histogram_bins < 128:
        p.error("val-split 必须在 (0,1)；histogram-bins 至少 128。")
    if args.output_dir is None:
        args.output_dir = ROOT / "int8_percentile" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return args


def make_session(path, threads, debug=False):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    if debug:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def clip_array(array, percentile, mode, axis=None):
    """权重使用精确 NumPy 百分位；axis 为输出通道轴，None 表示整张量。"""
    if not array.size or not np.isfinite(array).all():
        raise ValueError("权重为空或含 NaN/Inf。")
    reduce_axes = None if axis is None else tuple(i for i in range(array.ndim) if i != axis)
    if mode == "abs":
        threshold = np.percentile(np.abs(array), percentile, axis=reduce_axes, keepdims=True)
        lower, upper = -threshold, threshold
    else:
        tail = (100.0 - percentile) / 2.0
        lower, upper = np.percentile(array, [tail, 100.0 - tail],
                                     axis=reduce_axes, keepdims=True)
    lower, upper = lower.astype(array.dtype), upper.astype(array.dtype)
    clipped = np.clip(array, lower, upper)
    fraction = float(np.mean((array < lower) | (array > upper)) * 100)
    return clipped, lower, upper, fraction


def clip_graph_weights(source, destination, percentile, mode, per_channel):
    """仅截断实际参与 Conv/Gemm/MatMul 的常量权重，不截 bias、BN/LN 参数。

    在 ONNX 预处理/BN 折叠后执行，使阈值对应真正参与量化的卷积权重。
    每个消费者复制一份权重，避免共享 initializer 的轴定义不同或重复截断。
    """
    graph = onnx.load(str(source))
    initializers = {t.name: t for t in graph.graph.initializer}
    used_names = set(initializers)
    rows = []
    for index, node in enumerate(graph.graph.node):
        if node.op_type not in OPS or len(node.input) < 2 or node.input[1] not in initializers:
            continue  # attention 等动态矩阵的输入是激活，不作为常量权重处理
        tensor = initializers[node.input[1]]
        if tensor.data_type != onnx.TensorProto.FLOAT:
            continue
        weight = numpy_helper.to_array(tensor)
        if node.op_type == "Conv":
            axis = 0
        elif node.op_type == "Gemm":
            trans_b = next((a.i for a in node.attribute if a.name == "transB"), 0)
            axis = 0 if trans_b else 1
        else:
            axis = weight.ndim - 1
        selected_axis = axis if per_channel else None
        clipped, lower, upper, fraction = clip_array(weight, percentile, mode, selected_axis)
        new_name = f"{tensor.name}__percentile_{index}"
        while new_name in used_names:
            new_name += "_"
        used_names.add(new_name)
        graph.graph.initializer.append(numpy_helper.from_array(clipped, new_name))
        node.input[1] = new_name
        rows.append({"node": node.name, "original_weight": tensor.name,
                     "clipped_weight": new_name, "shape": list(weight.shape),
                     "output_channel_axis": selected_axis,
                     "lower": lower.ravel().tolist(), "upper": upper.ravel().tolist(),
                     "clipped_percent": fraction})
    # 删除已经没有消费者的原始 initializer，避免 FP32 原权重残留导致文件虚增。
    referenced = {name for node in graph.graph.node for name in node.input}
    referenced.update(t.name for t in graph.graph.input)
    referenced.update(t.name for t in graph.graph.output)
    kept = [t for t in graph.graph.initializer if t.name in referenced]
    del graph.graph.initializer[:]
    graph.graph.initializer.extend(kept)
    onnx.checker.check_model(graph)
    onnx.save(graph, str(destination))
    return rows


class OneImage(CalibrationDataReader):
    def __init__(self, images):
        self.feed = {"images": images.numpy()}

    def get_next(self):
        feed, self.feed = self.feed, None
        return feed


def calibrate(path, cache_path, data, indices, shape, args, mode):
    """每次只处理一张图片的激活，跨图片累计直方图而非保存所有激活。

    abs: ONNX Runtime symmetric=True => 对绝对值建立直方图。
    central: symmetric=False => 对原始有符号值建直方图，左右各截 (100-p)/2%。
    """
    with tempfile.TemporaryDirectory(prefix="calibration_", dir=cache_path.parent) as temp:
        augmented = Path(temp) / "augmented.onnx"
        calibrator = create_calibrator(
            path, OPS, augmented_model_path=str(augmented),
            calibrate_method=CalibrationMethod.Percentile,
            providers=["CPUExecutionProvider"],
            extra_options={"percentile": args.percentile, "symmetric": mode == "abs",
                           "num_bins": args.histogram_bins},
        )
        calibrator.infer_session = make_session(augmented, args.threads, debug=True)
        for count, (images, _) in enumerate(batches(data, indices, 1), 1):
            if tuple(images.shape[1:]) != shape or not torch.isfinite(images).all():
                raise ValueError(f"校准图片须为有限值且尺寸为 {shape}，实际 {tuple(images.shape[1:])}。")
            # ORT 每批打印直方图提示；隐藏重复提示，保留下面的进度输出。
            with contextlib.redirect_stdout(io.StringIO()):
                calibrator.collect_data(OneImage(images))
            if count % 50 == 0 or count == len(indices):
                log(f"  校准 {count}/{len(indices)} 张")
        ranges = calibrator.compute_data()
        rows = []
        for name, tensor in ranges.items():
            lower, upper = tensor.range_value
            lo, hi = float(lower), float(upper)
            if not np.isfinite([lo, hi]).all() or lo > hi:
                raise ValueError(f"无效校准范围 {name}: [{lo}, {hi}]")
            rows.append({"tensor": name, "lower": lo, "upper": hi})
        save_tensors_data(ranges, cache_path)
        write_json(cache_path.parent / "activation_ranges.json", rows)
        # Windows 必须释放 ORT 会话后才能删除临时 augmented.onnx。
        del calibrator, ranges
        gc.collect()


def predict(path, data, indices, shape, args):
    session = make_session(path, args.threads)
    predictions, truths = [], []
    seen = 0
    for images, labels in batches(data, indices, args.batch_size):
        if tuple(images.shape[1:]) != shape:
            raise ValueError(f"评估图片尺寸必须为 {shape}。")
        logits = session.run(None, {"images": images.numpy()})[0]
        if logits.ndim != 2 or not np.isfinite(logits).all():
            raise RuntimeError(f"{path} 推理输出无效。")
        predictions.append(logits.argmax(1))
        truths.append(labels.numpy())
        seen += len(images)
        if seen % 400 == 0 or seen == len(indices):
            log(f"  评估 {seen}/{len(indices)} 张")
    return np.concatenate(predictions), np.concatenate(truths)


def metrics(pred, baseline, truth, classes):
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(matrix, (truth, pred), 1)
    fp_accuracy = float(np.mean(baseline == truth) * 100)
    accuracy = float(np.mean(pred == truth) * 100)
    return {"samples": len(truth), "fp32_accuracy_percent": fp_accuracy,
            "int8_accuracy_percent": accuracy,
            "accuracy_drop_percentage_points": fp_accuracy - accuracy,
            "prediction_agreement_percent": float(np.mean(pred == baseline) * 100),
            "confusion_matrix": matrix.tolist(),
            "per_class_accuracy_percent": {
                c: float(matrix[i, i] / matrix[i].sum() * 100) if matrix[i].sum() else None
                for i, c in enumerate(classes)}}


def quantization_parameters(graph):
    """记录模型中实际生效的 scale/zero_point，便于核对校准范围和整数饱和边界。"""
    tensors = {t.name: numpy_helper.to_array(t) for t in graph.graph.initializer}
    rows = []
    for node in graph.graph.node:
        if node.op_type != "QuantizeLinear" or len(node.input) < 3:
            continue
        scale, zero = tensors[node.input[1]], tensors[node.input[2]]
        info = np.iinfo(zero.dtype)
        rows.append({"tensor": node.input[0], "dtype": str(zero.dtype),
                     "scale": scale.ravel().tolist(), "zero_point": zero.ravel().tolist(),
                     "effective_lower": ((info.min - zero.astype(np.float64)) * scale).ravel().tolist(),
                     "effective_upper": ((info.max - zero.astype(np.float64)) * scale).ravel().tolist()})
    return rows


def main():
    args = parse_args()
    if "calibration_cache_path" not in inspect.signature(quantize_static).parameters:
        raise RuntimeError("当前 ONNX Runtime 不支持校准缓存 API；请使用已检查的 1.31.0 环境。")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"输出目录非空，请另选新目录：{args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model, num_classes = build_model(args.weights)
    data, evaluation, calib_ids, eval_ids, full_evaluation = make_data(args, num_classes)
    example = data[calib_ids[0]][0].unsqueeze(0)
    shape = tuple(example.shape[1:])
    fp_path = args.output_dir / "model_fp32.onnx"
    pre_path = args.output_dir / "model_fp32_preprocessed.onnx"
    log(f"输出：{args.output_dir}\n模型：MobileViTv2 0.75 / ReLU / {num_classes} 类；输入 {shape}")
    native = export_model(model, example, fp_path)
    quant_pre_process(str(fp_path), str(pre_path), skip_symbolic_shape=True)
    for path in [fp_path, pre_path]:
        session = make_session(path, args.threads)
        np.testing.assert_allclose(session.run(None, {"images": example.numpy()})[0],
                                   native, rtol=1e-3, atol=1e-4,
                                   err_msg="ONNX 与原始 PyTorch 输出不一致，停止量化。")
        del session
    del model
    graph = onnx.load(str(pre_path))
    write_json(args.output_dir / "nodes.json",
               [{"name": n.name, "op_type": n.op_type} for n in graph.graph.node])
    del graph
    for name, dataset, ids in [("calibration", data, calib_ids), ("evaluation", evaluation, eval_ids)]:
        write_json(args.output_dir / f"{name}_samples.json",
                   [{"path": dataset.samples[i][0], "label": dataset.samples[i][1]} for i in ids])
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    write_json(args.output_dir / "run_config.json", {
        "configuration": config, "classes": data.classes, "input_shape_chw": shape,
        "full_evaluation_split": full_evaluation,
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "versions": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__},
        "scope": "Conv/Gemm/MatMul W8A8 静态 PTQ；其他算子可保持浮点；CPUExecutionProvider。",
    })
    log("评估原始 FP32 基线（所有组合共用，不提前截断原始模型权重）")
    baseline, truth = predict(fp_path, evaluation, eval_ids, shape, args)
    log(f"FP32 准确率：{np.mean(baseline == truth) * 100:.4f}%")
    combinations = ([(mode, weights) for mode in ["abs", "central"] for weights in [False, True]]
                    if args.compare_all else [(args.clip_mode, args.clip_weights)])
    summaries = []
    for mode, clip_weights in combinations:
        start = time.perf_counter()
        name = f"{mode}_{'activation_and_weight' if clip_weights else 'activation_only'}"
        directory = args.output_dir / name
        directory.mkdir()
        log(f"\n[{name}] Percentile={args.percentile}；权重截断={clip_weights}")
        source = pre_path
        if clip_weights:
            source = directory / "model_weight_clipped_fp32.onnx"
            weight_ranges = clip_graph_weights(pre_path, source, args.percentile, mode, args.per_channel)
            if not weight_ranges:
                raise RuntimeError("未找到可截断的常量权重，停止量化。")
            write_json(directory / "weight_ranges.json", weight_ranges)
        cache_path = directory / "calibration_cache.json"
        calibrate(source, cache_path, data, calib_ids, shape, args, mode)
        int_path = directory / "model_int8_qdq.onnx"
        quantize_static(
            source, int_path, calibration_cache_path=cache_path,
            calibrate_method=CalibrationMethod.Percentile, quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
            per_channel=args.per_channel, op_types_to_quantize=OPS,
            # abs 模式使用对称 scale，确保实际范围以 0 为中心；central 模式非对称。
            extra_options={"ActivationSymmetric": mode == "abs", "WeightSymmetric": True},
        )
        onnx.checker.check_model(str(int_path), full_check=True)
        quantized = onnx.load(str(int_path))
        int8_weights = sum(t.data_type == onnx.TensorProto.INT8 for t in quantized.graph.initializer)
        if not int8_weights or not any(n.op_type == "QuantizeLinear" for n in quantized.graph.node):
            raise RuntimeError("模型未生成预期 INT8 权重或静态激活 QDQ 边界。")
        write_json(directory / "quantization_parameters.json", quantization_parameters(quantized))
        operator_counts = dict(Counter(n.op_type for n in quantized.graph.node))
        del quantized
        pred, labels = predict(int_path, evaluation, eval_ids, shape, args)
        np.testing.assert_array_equal(labels, truth)
        result = metrics(pred, baseline, truth, data.classes)
        result.update({"configuration": name, "clip_mode": mode, "clip_weights": clip_weights,
                       "percentile": args.percentile, "weight_per_channel": args.per_channel,
                       "model_path": str(int_path.resolve()), "full_evaluation_split": full_evaluation,
                       "model_size_bytes": int_path.stat().st_size,
                       "onnx_operator_counts": operator_counts,
                       "elapsed_seconds": time.perf_counter() - start})
        write_json(directory / "report.json", result)
        row = {k: result[k] for k in ["configuration", "fp32_accuracy_percent", "int8_accuracy_percent",
                                      "accuracy_drop_percentage_points", "prediction_agreement_percent",
                                      "model_size_bytes", "elapsed_seconds", "model_path"]}
        summaries.append(row)
        # 每完成一个组合即保存结果，中断时已完成的比较不会丢失。
        with (args.output_dir / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(summaries)
        log(f"  FP32 {result['fp32_accuracy_percent']:.4f}% → INT8 {result['int8_accuracy_percent']:.4f}%"
            f"；下降 {result['accuracy_drop_percentage_points']:.4f} 个百分点")
        gc.collect()
    best = max(summaries, key=lambda r: r["int8_accuracy_percent"])
    write_json(args.output_dir / "best_configuration.json", {
        **best, "evaluated_configurations": len(summaries),
        "note": "仅代表本次评估集合中已运行配置的最佳结果；方案选择后的泛化精度需独立测试集验证。",
    })
    log(f"\n完成。本次已运行配置中最佳：{best['configuration']}，"
        f"INT8 {best['int8_accuracy_percent']:.4f}%\n比较表：{args.output_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()
