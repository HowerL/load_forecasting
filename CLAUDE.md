本文件是项目说明和协作指南，记录项目结构、主要流程、代码约定和关键配置。不要在这里记录某次运行得到的实验指标、图表数值或中间日志；这些内容以重新运行 notebook 生成的结果文件为准。

## 项目概述

本项目用于“基于迁移学习的校园建筑用能负荷预测研究”。整体思路是：对校园建筑逐小时电力负荷、天气和日历数据进行预处理，通过 VMD-DTW 相似性分析辅助选择源域建筑，再构造源域/目标域数据集，比较 LSTM 迁移学习策略在目标域小样本场景下的表现，并在同一数据流程后续补充不同预测步长下的多步预测对比。

基本流程：

```text
data_preprocess.ipynb
→ similarity_analysis.ipynb
→ prepare_transfer_data.ipynb
→ transfer_learning.ipynb
→ multistep_comparison.ipynb
→ visualization.ipynb
```

## Agent 运行环境约定

本项目使用仓库内虚拟环境。Agent 如需通过终端运行 `python`、`pip`、`jupyter`、`pytest` 或依赖 Python 解释器的命令，必须先按当前终端类型调用 `.venv/Scripts` 中对应的激活脚本，并在同一条终端命令会话内继续执行后续命令，避免因新会话未继承环境而出现找不到 `python` 或误用系统 Python 的问题。

- Git Bash / Bash：`source .venv/Scripts/activate`
- PowerShell：`. .venv/Scripts/Activate.ps1`
- CMD：`call .venv/Scripts/activate.bat`

如果只需要执行单个 Python 模块命令，也可以直接使用虚拟环境解释器，例如 `.venv/Scripts/python.exe -m pytest` 或 `.venv/Scripts/python.exe -m pip install ...`。在受限 PowerShell 或自动化沙箱中，`Activate.ps1` 可能无法 dot-source；此时不要继续调用裸 `python`，应直接使用 `.venv/Scripts/python.exe`。不要依赖未激活环境中的 `python`、`python3`、`py` 或全局 `pip`。

## Notebook 职责

| Notebook | 作用                                                 |
| --- |----------------------------------------------------|
| `data_preprocess.ipynb` | 处理原始建筑负荷、天气和日历数据，生成各建筑特征文件                         |
| `similarity_analysis.ipynb` | 对建筑负荷模式做 VMD-DTW 相似性排序，辅助选择源域/目标域组合                |
| `prepare_transfer_data.ipynb` | 根据 `src/config.py` 中的源域/目标域配置，生成标准化迁移学习数据          |
| `transfer_learning.ipynb` | 先用目标域验证集网格搜索 LSTM 结构参数，再训练源域模型，运行源域直测、目标域训练和三种迁移学习策略，并输出多 seed 明细、均值汇总与配对检验结果  |
| `multistep_comparison.ipynb` | 迁移学习后的多步预测对比环节，验证不同 `HORIZON` 下目标域训练与全层微调的多步预测表现 |
| `visualization.ipynb` | 汇总生成负荷曲线、相关性分析和迁移学习结果图                             |

## 关键配置

共享配置位于 `src/config.py`：

- `BUILDINGS`：参与实验的建筑列表
- `SOURCE_BUILDING` / `TARGET_BUILDING`：当前迁移学习实验使用的源域和目标域，由人工根据相似性分析结果配置
- `TARGET_SAMPLE_START` / `TARGET_SAMPLE_END`：目标域小样本窗口
- `BASE_DIR` / `BUILDINGS_DIR` / `SCALER_DIR` / `MODEL_DIR` / `FIGURES_DIR`：输出路径
- `TARGET_COL` / `TIME_COL`：负荷目标列和时间列
- `SEED`：用于固定生成论文展示图表和同名模型文件的随机种子；统计汇总时与 `REPEAT_SEEDS` 中的 seed 一起纳入多 seed 重复实验

`src/split_standardize.py` 中的 `TRAIN_RATIO = 0.70` 通过 `src/__init__.py` 暴露为公共 API，供 notebook 对齐数据切分和相似性选择窗口。`VAL_RATIO = 0.15` 仍是 `split_standardize.py` 内部默认参数。

模型超参数如 `LOOKBACK`、`HORIZON`、`BATCH_SIZE`、`EPOCHS` 以及 LSTM 结构候选网格保持在各 notebook 内部定义，便于实验调整。

## 数据与特征流程

`data_preprocess.ipynb`：

- 将原始累计电量转换为小时负荷。
- 使用滑动 MAD 检测异常值，异常值转为缺失后再填充。
- 缺失填充优先使用前 7 天同小时均值，不足时回退到前 14 天同小时均值，最后用时间插值和前向填充兜底。
- 天气连续特征使用全量天气数据拟合 `weather_scaler.joblib`；天气类别特征使用 one-hot 编码。
- 时间特征包括 `hour`、`day_of_week`、`month` 的 sin/cos 编码和 `is_holiday`。

主要输入：

- `data/buildings/{建筑名}.CSV`
- `data/天气.CSV`
- `data/2024日历.csv`

主要输出：

- `data/buildings/{建筑名}_预处理后.csv`
- `data/buildings/{建筑名}_特征.csv`
- `data/天气_预处理后.csv`
- `data/scalers/weather_scaler.joblib`

## 相似性分析约定

`similarity_analysis.ipynb` 对每个建筑的负荷独立标准化后计算 VMD-DTW，源域选择排序以 `加权DTW` 升序为主判据，距离越小表示两栋建筑在 VMD 分解后的主要模态上越接近。不再使用百分比形式的辅助指标作为结果指标。

为避免评估泄漏，相似性分析不使用完整目标小样本窗口，而是在 `TARGET_SAMPLE_START <= timestamp < TARGET_SAMPLE_END` 内取所有建筑公共时间索引的前 `TRAIN_RATIO` 部分作为相似性选择窗口。VMD 参数优化和成对 VMD-DTW 计算都使用这个窗口。该 notebook 不导入 `TARGET_BUILDING`，因为源域和目标域是在相似性排序后由人工确认。

VMD 参数采用建筑对级别的联合优化。每个建筑对共用同一组 `K` 和 `alpha`，以保证两栋建筑分解得到的 IMF 具有可比性。当前搜索范围为 `K ∈ [2, 6]`、`alpha ∈ [500, 5000]`；`K` 在适应度函数调用处由连续候选值四舍五入并裁剪为整数，`alpha` 作为连续变量搜索。优化算法使用 `scipy.optimize.differential_evolution`，目标是最小化两个建筑 IMF 经验熵适应度的均值。

差分进化优化直接使用从 `src.config` 导入的全局 `SEED` 作为随机种子，用于保证同一数据、同一窗口和同一搜索范围下的 VMD 参数优化结果可复现。

经验熵适应度的计算流程如下：

1. 给定候选参数 `(K, alpha)`，分别对建筑 A 和建筑 B 的标准化负荷序列做 VMD，得到 `K` 个 IMF。
2. 对每个 IMF 用直方图估计经验概率分布，当前使用 `VMD_ENTROPY_BINS = 30`。若第 `i` 个 IMF 的经验概率为 `p_ij`，则经验熵为 `H_i = -Σ p_ij log(p_ij)`。
3. 按 IMF 能量占比计算权重，`w_i = Σ u_i(t)^2 / Σ_i Σ_t u_i(t)^2`。能量越大的 IMF 对总适应度影响越大。
4. 单栋建筑的 VMD 适应度为 `F = Σ w_i H_i`。建筑对适应度为 `F_pair = (F_A + F_B) / 2`。
5. 差分进化迭代搜索使 `F_pair` 最小的 `(K, alpha)`，再用该参数组合执行最终 VMD-DTW。

这一步解决了“只优化 K、固定 alpha”的问题：`alpha` 不再是经验给定常数，而是和 `K` 一起由适应度函数驱动搜索。经验熵越小，表示 IMF 的经验概率分布越集中、随机复杂程度越低；用它作为目标函数，可以从数据本身选择更能产生清晰模态分解的 VMD 参数组合。`加权DTW` 仍只负责建筑对排序，经验熵适应度只负责 VMD 参数选择，两者职责分开。

由于适应度定义为建筑对均值 `F_pair = (F_A + F_B) / 2`，参数优化不能沿用旧版本“每栋建筑预计算一个 K，再对建筑对取较大值”的方式。当前可预计算或缓存的合理对象是建筑对级别参数，即“建筑A-建筑B → K、alpha、经验熵适应度”。只有当输入数据、相似性选择窗口、搜索范围、随机种子和适应度函数都未变化时，才可以复用缓存结果。

相似性结果输出：

- `data/相似性分析汇总.csv`
- `data/vmd_parameter_optimization.csv`
- `data/相似性分析完整结果.json`
- `data/figures/原始负荷对比.png`
- `data/figures/IMF{n}对比.png`

`相似性分析汇总.csv` 保存 `K`、`alpha`、经验熵适应度、源域经验熵、目标域经验熵、`加权DTW` 和 `原始DTW`。notebook 控制台表格只打印用于阅读和排序的核心列；源域经验熵和目标域经验熵只保存到文件中，不在控制台表格展示。

如果论文需要展示“源域选择有效性”，采用轻量手动对比：分别选择 `加权DTW` 最小和最大的两组建筑对，人工修改 `SOURCE_BUILDING` / `TARGET_BUILDING` 后重跑数据准备和迁移学习，计算：

```text
PIR = (target_train_MAE - transfer_MAE) / target_train_MAE * 100
```

只有两组对比时不报告 Spearman/Kendall 相关性。

## 数据切分与标准化

`prepare_transfer_data.ipynb` 使用 `src/config.py` 指定的源域和目标域：

- 源域：使用全量特征数据，只标准化负荷，不做 train/val/test 切分。
- 目标域：截取 `TARGET_SAMPLE_START` 到 `TARGET_SAMPLE_END` 的小样本窗口，按 70%/15%/15% 划分训练、验证、测试。
- 天气特征已在预处理阶段标准化；迁移学习阶段只对负荷列做域内标准化。
- 目标域验证集和测试集复用目标训练集拟合出的负荷 scaler。

输出：

- `data/source_train_std.csv`
- `data/target_train_std.csv`
- `data/target_val_std.csv`
- `data/target_test_std.csv`
- `data/scalers/source_load_scaler.joblib`
- `data/scalers/target_load_scaler.joblib`

## 模型与训练

公共模型在 `src/models.py`：

- `LSTMPredictor`：LSTM + LeakyReLU + 线性输出层，`hidden_size` 控制隐藏层单元数，`num_layers` 控制 LSTM 层数，`horizon` 控制输出维度；模型固定使用 LeakyReLU 激活函数，不再提供激活开关。
- `FeatureExtractorRegressor`：复用并冻结预训练 LSTM，根据预训练 LSTM 的实际 `hidden_size` 构造新的两层回归头，层间固定使用 LeakyReLU。

公共训练工具在 `src/training.py`：

- `set_seed()`
- `LoadDataset`
- `EarlyStopping`
- `run_epoch()`
- `predict()`
- `evaluate()`：返回 `MAE`、`RMSE`、`CV-RMSE`、`MAPE`、`R2`

序列构造由 `src/preprocessing.py:create_sequences()` 完成，要求时间戳逐小时连续；输出形状为 `X=(N, lookback, feature_count)`，`y=(N, horizon)`。`transfer_learning.ipynb` 的数据流保持为：DataFrame → `create_sequences()` 生成 ndarray → `LoadDataset` 包装为 `torch_datasets`；各策略函数内部按需构造 `DataLoader`。

完整训练循环保留在 notebook 中，不下沉到 `src/`。`src/` 只保留可复用的模型、数据集、指标和单 epoch 训练/验证函数。

`transfer_learning.ipynb` 中多 seed 执行编排直接写在 `[*REPEAT_SEEDS, SEED]` 循环体中；多 seed 均值汇总和相对 `目标域训练` 的配对统计检验直接写在“实验结果汇总”代码单元中。除 `mean_ci()` 这类会重复使用的小型统计辅助函数外，不为只调用一次的执行编排或汇总逻辑额外定义函数。

`transfer_learning.ipynb` 在正式多 seed 迁移学习实验前，先以目标域验证集 `MAE` 最小为目标，对 `hidden_size=[64, 128, 256]` 和 `num_layers=[1, 2]` 做网格搜索。测试集不参与参数选择。搜索得到的 `HIDDEN_SIZE` 和 `NUM_LAYERS` 被后续所有策略统一复用，保证策略比较不受模型结构差异影响。

`transfer_learning.ipynb` 当前包含的策略：

1. `源域直测（Source-only）`：只在源域训练，不在目标域迁移，直接在目标测试集预测。
2. `目标域训练（Target-only）`：不在源域训练，只在目标域训练集上从零训练模型，作为主参照。
3. `全层微调（Full fine-tuning）`：加载源域训练权重后在目标域微调全部层。
4. `固定时序表征微调（Fixed temporal representation fine-tuning）`：固定 LSTM 时序表征层，仅微调输出映射层。
5. `固定特征回归（Fixed-feature regression）`：将预训练 LSTM 作为固定特征提取器，重新训练目标域回归映射层。

当前训练设置：

- 源域训练与 `目标域训练`：`lr=1e-3`，`weight_decay=1e-4`
- `全层微调`：`lr=1e-4`，`weight_decay=1e-4`
- `固定时序表征微调` 与 `固定特征回归`：`lr=1e-3`，`weight_decay=1e-4`
- 各训练函数内部创建 `EarlyStopping(patience=5, restore_best_weights=True)` 并传入 `fit_model()`，便于后续按策略单独调整
- `ReduceLROnPlateau(factor=0.5, patience=4)`
- 多 seed：实际执行顺序直接使用 `[*REPEAT_SEEDS, SEED]` 一起参与统计汇总

每个 seed 运行都会保存模型和图表；由于保存路径固定，后执行的 seed 会覆盖同名模型和图表。`SEED` 排在执行列表最后，仅用于固定同名模型和预测图的最终保存版本；定量比较以所有 seed 的平均结果、置信区间和配对检验为主要依据。

## 结果文件

迁移学习输出：

- `models/pretrained_source.pt`
- `models/transfer_target_only.pt`
- `models/transfer_full_fine_tuning.pt`
- `models/transfer_fixed_temporal_representation_fine_tuning.pt`
- `models/transfer_fixed_feature_regression.pt`
- `data/transfer_learning_lstm_grid_search_results.csv`
- `data/transfer_learning_strategy_mean_metrics.csv`
- `data/transfer_learning_per_seed_metrics.csv`
- `data/transfer_learning_strategy_metric_stats.csv`
- `data/transfer_learning_paired_tests_vs_target.csv`

`transfer_learning_strategy_mean_metrics.csv` 由 `transfer_learning.ipynb` 生成，保存各策略在所有 seed 上的平均 `MAE`、`RMSE`、`CV-RMSE`、`MAPE` 和 `R2`，供后续可视化和论文结果表使用。

迁移学习训练与预测图使用中文文件名保存，包括 `源域预训练_损失曲线.png`、`源域直测_预测对比.png`、`目标域训练_损失曲线.png`、`目标域训练_预测对比.png`、`全层微调_损失曲线.png`、`全层微调_预测对比.png`、`固定时序表征微调_损失曲线.png`、`固定时序表征微调_预测对比.png`、`固定特征回归_损失曲线.png` 和 `固定特征回归_预测对比.png`。

多步预测输出：

- `models/multistep_source_h{1,3,6}.pt`
- `data/multistep_overall_metrics.csv`
- `data/multistep_per_step_results.csv`
- `data/figures/逐步误差变化_H{3,6}.png`

`multistep_comparison.ipynb` 只保留两种结果策略：`目标域训练（Target-only）` 和 `全层微调（Full fine-tuning）`。源域训练只用于为全层微调提供初始权重，不作为结果策略展示。该 notebook 按当前 `HORIZON` 在循环内构造序列和 `torch_datasets`，各训练步骤内部自行构造 `DataLoader`、`EarlyStopping` 和学习率调度器。多步实验复用 `transfer_learning_lstm_grid_search_results.csv` 中按目标域验证集选出的 `hidden_size` 和 `num_layers`，并保持源域训练、目标域训练、全层微调三个同名阶段的优化器、学习率、权重衰减、调度器和早停设置与 `transfer_learning.ipynb` 一致。训练过程中复用公共可视化函数在 notebook 输出区展示 loss 曲线和预测结果图，不保存这些训练过程图。多步实验只保存所有预测步合并计算的整体指标表，不输出整体指标对比图；`H=1` 的结果保留在表格和 CSV 中但不输出逐步误差图，`H=3/6` 按 `HORIZON` 分别输出独立图片，每张图只比较同一 `HORIZON` 下的目标域训练和全层微调，不再单独保存最近一步精度对比表。

主要保存图表位于 `data/figures/`，包括相似性分析图、迁移学习训练/预测图、迁移学习指标对比图和多步预测逐步误差变化图。`visualization.ipynb` 中的负荷曲线和特征相关性分析主要在 notebook 内展示；除迁移学习指标对比图外不另存。具体文件以各 notebook 的保存路径为准。

## 维护约定

- 以代码和 notebook 为准维护本文档；本文档不作为 changelog 使用。
- 不记录某次运行的指标数值、最佳 epoch、排序结论或图表解释。
- 修改实验逻辑后，同步更新相关 notebook、`src/` 公共接口说明和本文件中的结构性描述。
