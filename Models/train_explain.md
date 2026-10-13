## 按驾驶员划分训练与验证集

* 使用 `driver_imgs_list.csv` 的 `(classname, img)` 对应 `train_p` 图片，以 `subject` 整组划分。
* 目标按图片数量训练:验证约为 8:2；固定 `SEED=42`，只根据样本数和类别分布选择组合，不使用模型验证结果挑选驾驶员。运行时检查驾驶员无交集、图片完整且不重复、两边覆盖所有类别。
* `train_p` 的预处理图片继续使用，文件名或类别无法对应 CSV 时直接报错。驾驶员名单、实际比例及样本映射摘要随检查点保存。
* 首次运行保持 `RESUME=False`，从 ImageNet 权重重新训练。旧的按图片随机划分检查点及无法核对划分来源的纯权重文件不能用于本次续训；后续可使用相同驾驶员划分的新检查点。
* 最佳模型与每轮检查点名称增加 `driver_split`，避免覆盖之前随机图片划分的训练结果。

---

# MobileViTv2-0.75x 微调训练（ImageNet-1k 预训练 → 10 类，激活函数换为 ReLU）

**按要求做的调整**

1. 宽度 `WIDTH_MULTIPLIER = 0.75`，与纯权重文件 `mobilevitv2-0.75.pt`（ImageNet-1k，269 个张量，2,877,698 个参数）一致；
2. 激活函数由 `swish`(SiLU) 改为 `relu`：`build_opts` 里同时设置
   `model.activation.name` 与 `model.classification.activation.name`。ReLU / Swish 都没有可学习参数，
   所以 state_dict 的键完全不变（已实测：换激活后 269 个键一一对应），ImageNet 权重可以原样加载；
3. 类别数由数据集自动读取（`train_p` 下 10 个类别目录 c0~c9，共 22,424 张 256×256 RGB 图）；
4. 用 `PRETRAINED_WEIGHTS` 初始化：加载除 1000 类分类头以外的全部 backbone 权重（267/269），
   10 类分类头按 cvnets 的 `trunc_normal` 重新初始化。

**修复的原脚本问题**

| 问题 | 说明 |
| --- | --- |
| 预训练权重根本加载不上 | 改成 10 类后 `classifier.1.weight` 由 (1000,384) 变 (10,384)，实测 `load_state_dict` 无论 `strict` 与否都会抛 `RuntimeError`；现在先按形状筛出可加载的键，再 `strict=True` 加载，并校验"除分类头外没有别的键被跳过" |
| 余弦退火实际失效 | 原 `T_MAX_ITERS=1000` < 预热步数 3366，`CosineAnnealingLR(T_max=1000-3366)` 收到负数；现在 `T_MAX_ITERS=None` 时自动取 `EPOCHS × 每轮步数`，不合法直接报错 |
| EMA 起点错误 | 原脚本在加载权重**之前**创建 EMA，且"纯权重"分支没有重建 EMA，导致评估/保存用的 EMA 从随机初始化开始；现在权重加载或恢复之后再创建 EMA |
| MixUp/CutMix 分支缺少标签平滑 | 混增强分支用裸 `F.cross_entropy`，与主 `criterion` 的 `label_smoothing` 不一致 |
| 训练准确率恒为 -1 | 增强开启时直接返回 -1；现在用"未增强的 batch"统计并明确标注为近似值 |
| 检查点信息不足 | 检查点现在带 `config`（类别名、宽度、激活、T_max、数据集路径等），推理端可据此交叉校验类别顺序 |
| 其它 | 重复 import cvnets、`t_max_value` 重复计算、`warmup_steps` 为 0 时除零、断点续训与保存路径撞车、可能覆盖 ImageNet 预训练权重 |

**运行顺序**：依次运行下面 3 个代码单元即可；`main()` 在最后一个单元里，改超参后重跑该单元。

数据集：`C:\transformerFPGA\MobileViT\Dataset\Driver\train_p`。原图已经是 256×256（能被 32 整除），
因此**不做 Resize**，训练/验证/推理三处预处理保持一致（Resize(可选) → ToTensor → ImageNet 归一化）。

---

## 本次修订（按代码审阅意见）

审阅提出的问题逐条核对后确认成立，已全部修正：

| # | 问题 | 修正 |
| --- | --- | --- |
| 1 | **EMA 检查点不能正确续训**：`model_state_dict` 里存的是 EMA 权重，而 `optimizer_state_dict` 来自原始模型，恢复时把 EMA 参数装进训练模型再套原优化器状态，两者不对应 | 统一走 `build_checkpoint()`：`model_state_dict` 存**原始权重**（续训用，与优化器状态配套），`ema_model_state_dict` 存 **EMA 权重**（推理用），并写入 `inference_weights` 指明推荐用哪一份；续训按 `model_state_dict` 恢复，EMA 在恢复后重新播种。想从 EMA 续训可设 `RESUME_FROM_EMA=True`（会明确告警） |
| 2 | **追加训练可能让学习率回升**：`CosineAnnealingLR` 在步数超过 `T_max` 后 cos 项重新变大，续训沿用已走完的调度器就会看到学习率上涨 | 换成单个 `LambdaLR`（`make_lr_lambda`），退火进度用 `min(1.0, …)` 夹住，超过周期后恒为 `ETA_MIN`；恢复时若发现已越过周期会明确提示（要重新退火就设 `RESUME_RESET_SCHEDULER=True`） |
| 3 | **重置调度器后首个优化步跳过预热起点**：先建调度器再把 `lr` 手动改成 `target_lr`，第一步直接用满学习率 | 先写好基准学习率与追加步数再建调度器，保留调度器初始化算出的当前学习率（= 预热起点）；重置周期改用 `ADDITIONAL_EPOCHS × 每轮步数`（原来错用 `EPOCHS`）；预热步数放不下时自动收缩到周期的 10% 并提示 |
| 4 | **只开一种增强时概率参数失效**：`elif mixup_enable: apply_mixup = True` 会让 `MIXUP_PROB=0.01` 形同虚设 | 单开也按各自概率抽样；`main()` 校验概率范围，并提示 `MIXUP_PROB + CUTMIX_PROB > 1` 的情况 |
| 5 | **梯度累积最后一组不足时梯度被多除** | 按当前累积组实际 batch 数归一化：`group_size = min(grad_accum_steps, num_batches - group_start)` |
| 6 | **"最佳准确率"与保存的最佳模型可能不是同一轮**：按最低验证损失保存，`best_val_acc` 却是历史最高准确率 | 检查点同时记录本轮 `val_loss`/`val_acc` 与判据/历史值 `best_loss`/`best_epoch`/`best_val_acc`，保存与收尾汇总都把两者分开打印 |
| 7 | **续训前仍然加载预训练文件**：白做一次加载，且预训练文件缺失会让续训直接失败 | `RESUME=True` 时跳过预训练初始化并说明原因 |

顺带清理的冗余：检查点字段只在 `build_checkpoint()` 里构造（早停保存与每轮保存共用）；续训状态统一走 `EarlyStopping.load_state()`；EMA 去掉未使用的 `*args/**kwargs`，`ema_momentum` 类型改为 `float`；混合增强损失直接用 `criterion`（不再单独传 `label_smoothing`）；类别计数改用 `collections.Counter`；删掉不再使用的导入（`Tuple`、`torch.nn.functional`）。

> `t.txt` 是本文件后两个代码单元的拼接，缺少第 1 个单元的导入以及 `build_opts` / `build_transform` / `load_weights` 等定义，所以它单独无法运行；在笔记本里按单元顺序执行即可。

> 检查点体积：`CHECKPOINT_INCLUDE_EMA=True` 时每个检查点同时含原始权重与 EMA 权重（约 40MB／轮，30 轮约 1.2GB）。只想留最佳模型就把 `CHECKPOINT_EVERY_EPOCH` 设为 `False`；关掉 `CHECKPOINT_INCLUDE_EMA` 只省掉 EMA 那一份权重（优化器状态仍在，实际节省不到一半）。

---

## 第二轮修订（按第二次审阅意见）

| # | 问题 | 修正 |
| --- | --- | --- |
| 1 | **续训没有恢复 EMA**：检查点里存了 `ema_model_state_dict`，恢复时却只用原始权重重建 EMA，历史滑动平均被丢弃，验证指标会突变 | 续训时把 EMA 权重**严格加载回 EMA 模型**（形状/键不匹配会告警并退回重建），验证指标连续 |
| 2 | **缺失的主干权重发现不了**：`model_state.update(loadable)` 先把所有键补齐，`strict=True` 就失去了检查"文件缺键"的能力（只给分类头也能通过） | `load_weights()` 改为双向检查：文件**缺少**模型需要的非分类头参数 → 报错；文件里有模型用不上的非分类头键 → 报错。局部测试：只给分类头、缺一个主干键都会报错，完整 10 类权重 269 个全部加载 |
| 3 | **恢复调度器状态 ≠ 恢复原曲线**：`make_lr_lambda` 的闭包参数不会被 `LambdaLR.state_dict()` 保存，配置一变曲线就变 | 检查点新增 `scheduler_config`（周期、预热步数、基准/起点/终点学习率、每轮步数）；续训时**先按保存的配置重建曲线，再加载调度器状态**。局部测试：恢复后 20 步与原曲线逐步一致，按"当前错误配置"恢复则偏到别的值 |
| 4 | **覆盖基准学习率会把起点/终点等比缩放**：原来返回"相对 `MAX_LR` 的倍率"，基准换 1e-4 后起点变 5e-7、终点 5e-11 | 改为**绝对学习率语义**：预热从 `WARMUP_START_LR` 升到基准，再余弦降到 `ETA_MIN`，与基准无关。局部测试：基准 1e-4 时起点 1e-6、峰值 1e-4、终点与周期后都是 1e-10 |
| 5 | **"最佳轮准确率"可能配错轮次**：`best_epoch` 来自最低损失轮，`val_acc` 却来自当前轮 | 新增 `best_loss_val_acc`（最低损失**那一轮**的准确率）并随检查点保存/恢复（旧格式回退到 `val_acc`）。局部测试：第 3 轮最佳、第 10 轮更差时三者互不串台 |
| 6 | **续训后不保证保存路径里真有最佳模型**：损失一直没改善就不写文件，收尾却报告"最佳模型"并去加载（可能不存在或读到旧文件） | 新增 `EarlyStopping.saved_this_run`：区分"历史最佳"与"本次是否写入"；没写入就明确报告并跳过收尾自检；保存路径已存在时先提示；另加 `RESUME_RESET_BEST_LOSS=True` 让追加训练重新判定最佳并写入新文件 |
| 7 | **关闭 `CHECKPOINT_INCLUDE_EMA` 时验证指标与保存权重不对应** | 最佳模型**始终**保存"验证时用的那份权重"（EMA 开启时含 EMA）；该开关只影响每轮检查点 |
| 8 | **梯度累积未按样本数加权**：尾组小 batch 与整批等权 | 组内按"该 batch 样本数 / 组内总样本数"加权（`batch_sample_counts()`），组内权重和为 1；`GRAD_ACCUM_STEPS=1` 时与原来完全一致。局部测试：30 样本、批 8、累积 2 → 尾组权重 8/14 与 6/14 |
| 9 | **`RESIZE=256` 不能保证尺寸一致**：整数只固定短边 | 文档与提示统一改成"要固定尺寸请写 `(256, 256)`"，`build_transform()` 对整数写法给出提示 |

冗余清理：删掉只赋值不读取的 `last_val_acc`；`include_ema` 不再层层传递（最佳模型保存恒为 True）；`rand_bbox` 的 `W/H` 命名改为 `row/col`（行为不变）；`run_meta` 记录**实际使用**的调度配置，并保留配置原始值便于对照。

> 以上 9 项都做了局部逻辑测试（不读数据、不训练、不用 GPU，共 9 项全部通过）；语法与静态检查（JSON、逐单元 compile、pyflakes）同样通过。

---

## 第三轮修订（按第三次审阅意见）

| # | 问题 | 修正 |
| --- | --- | --- |
| 1 | **调度配置恢复不完整**：保存了 `start_lr`/`min_lr`，恢复时却只读周期与基准，改了这两个配置续训曲线就变 | 恢复时把 `start_lr` / `min_lr` 一起读回并显式传给 `build_scheduler()`；只有主动 `RESUME_RESET_SCHEDULER=True` 时才使用当前配置 |
| 2 | **纯权重续训忽略 `RESUME_LR` / `RESUME_RESET_SCHEDULER`**，且退火周期仍按 `EPOCHS` 算，与实际只练 `ADDITIONAL_EPOCHS` 轮不匹配 | 把"本次学习率与退火周期"的定稿逻辑从完整检查点分支里提出来，两种续训模式统一处理；纯权重或主动重置时，周期 = `ADDITIONAL_EPOCHS × 每轮步数`（显式设了 `T_MAX_ITERS` 则优先用它） |
| 3 | **"历史最佳仍在续训文件里"不一定成立**（第 30 轮的检查点只有第 30 轮权重） | 检查点新增 `best_checkpoint_path`，记录历史最佳权重当时写到了哪个文件；续训与收尾报告改为如实引用它（并检查文件是否仍在），不再声称最佳权重在 `RESUME_PATH` 里 |
| 4 | **旧格式 `best_loss_val_acc` 兜底不准**：直接用 `val_acc` 会把当前轮准确率算到最佳轮 | 只有 `epoch == best_epoch` 时才允许用 `val_acc` 兜底，否则记为未知（报告显示"未知"） |
| 5 | **"已有文件不会被覆盖"的提示与实际行为不符** | 按实情分两种提示：续训且未重置最佳 → "只有更好才会覆盖"；从零开始 → "基准 val_loss=inf，第一轮就会覆盖" |
| 6 | **恢复失败后的回退没有真正重建对象** | 调度器恢复失败 → 重新 `build_scheduler()`；EMA 恢复失败 → 重新 `EMA(model, ...)`，回退状态确定 |
| 7 | 调度器被创建两次（创建后又因续训再建一次） | 先把配置定稿，再**只创建一次**调度器 |
| 8 | 组内样本总数被重复计算 | 同一个累积组只求一次，进组时缓存 |
| 9 | 注释不准确（采样前提 / "等价于大 batch"） | 改为"默认批处理方式 + `drop_last=False`，`shuffle` 无关"；并注明梯度累积**不等于**真正的大 batch（模型含 BN/Dropout，两者不会逐位一致） |
| 10 | `GRAD_ACCUM_STEPS` 未校验；`WARMUP_EPOCHS=0` 被强制成"至少预热 1 步" | 新增 `GRAD_ACCUM_STEPS` 必须为 ≥1 整数的校验；预热步数改为 `int(round(WARMUP_EPOCHS × 每轮步数))`，`0` 真的是 0 步；`Counter` 导入移到单元 1 |

### 实跑记录（8 张 256×256 图 / 2 类 / 各 1 轮）

* **运行 A（全新训练）**：`预热步数: 0（WARMUP_EPOCHS=0，不预热）` → 第 1 步 lr = `2.000e-04`（就是基准值，确实没预热）；训练 3 步、验证 2 张、保存 `best.pth` 与 `ckpt_1.pth`；收尾自检对 `model_state_dict` 与 `ema_model_state_dict` 都严格加载通过。
* **运行 B（续训）**：跳过预训练初始化 → `采用检查点保存的调度配置: 周期 3 步, 预热 0 步, 基准 0.0002, 起点 1e-06, 终点 1e-10` → `历史最佳权重当时保存在: ...best.pth（文件仍在）` → `已恢复调度器状态: 第 3 步, 当前学习率 1.000e-10`，并提示"已越过退火周期，学习率不会回升" → `EMA 已从检查点恢复` → 正常训练第 2 轮、保存新检查点、自检通过。

---

## 第四轮修订（按第四次审阅意见）

| # | 问题 | 修正 |
| --- | --- | --- |
| 1 | **`best_checkpoint_path` 记的是本次输出路径，而不是历史最佳的实际位置**：续训一直没改善时，每轮检查点会谎称历史最佳在本次的 `SAVE_PATH`（该文件可能根本不存在） | 这个字段改由 `EarlyStopping` 自己维护：`load_state()` 时**继承**检查点里的值，只有**真正写出**新的最佳模型时才更新为本次路径；`build_checkpoint()` 与所有报告都改读 `early_stopping.best_checkpoint_path` |
| 2 | **"周期 > 预热"的校验发生在最终配置定稿之前**，会用全新训练的 `EPOCHS/WARMUP_EPOCHS` 误拦合法的续训（例如 `EPOCHS=2`、`WARMUP_EPOCHS=3`，但检查点里的周期原本合法） | 基础检查（周期为正、预热非负）留在原处；"周期 > 预热"移到**最终调度配置定稿之后、创建调度器之前**判断 |
| 3 | **极短追加训练可能出现"预热 = 整个周期"**（周期 1 步、预热 1 步时，学习率走完这一步才刚到峰值，永远退不到终点） | 统一保证 `0 <= 预热 < 周期`：续训时自动收缩预热并提示；全新训练的配置不合理时直接报错；`make_lr_lambda()` 内部也强制该不变式 |

顺带清理：`config['scheduler']` 与顶层 `scheduler_config` 重复 → 只保留顶层作为正式字段（`config` 里两项注明"按超参推算的原值，仅用于对照"）；设置学习率的循环抽成 `reset_scheduler()`，创建与异常回退共用；注明 `LR` 是唯一入口（`MAX_LR` 只是峰值别名）；精简若干描述旧版错误的注释。**`build_opts` 的选项一个都没删**（必须与所装 cvnets 版本对应）。

### 实跑记录（8 张 256×256 图 / 2 类）

* **运行 A（全新训练 1 轮）**：`预热步数: 2`、周期 3 步（`WARMUP_EPOCHS=0.5`）；保存 `best.pth` 与 `ckpt_1.pth`；检查点里 `scheduler_config` 记录了起点/终点/预热/周期，`best_checkpoint_path` = 本次 `best.pth`。
* **运行 B（换新输出路径续训 1 轮）**：`SAVE_PATH=new_best.pth`、从 `ckpt_1.pth` 续训 → 结束后核对不变式：`ckpt_resume_2.pth['best_checkpoint_path']` 等于**真正写出最佳权重的那个文件**（本次有新最佳 → `new_best.pth`；没改善 → 仍是 A 的 `best.pth`），不再指向未生成的文件。
* **运行 C（极短追加：周期 = 1 步）**：`BATCH_SIZE=6`（每轮只有 1 个优化步）+ `RESUME_RESET_SCHEDULER=True` + `ADDITIONAL_EPOCHS=1` → 打印 `预热步数 1 不小于退火周期 1，已收缩为 0`，检查点里 `warmup_steps=0, horizon=1`，第 1 步学习率就是基准值 `2.000e-04`（不再停在峰值不动）。
* **运行 D（全新训练配置不合理）**：`EPOCHS=1` + `WARMUP_EPOCHS=1`（周期=预热=每轮步数）→ 直接报错并提示"把 WARMUP_EPOCHS 调小或 EPOCHS/T_MAX_ITERS 调大"，不会再生成一条"整段停在峰值"的曲线。