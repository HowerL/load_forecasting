本文件是项目说明和协作指南，记录项目结构、主要流程、代码约定和关键配置。不要在这里记录某次运行得到的实验指标、图表数值或中间日志；这些内容以重新运行 notebook 生成的结果文件为准。

## 项目概述

本项目用于“基于迁移学习的校园建筑用能负荷预测研究”。整体思路是：对校园建筑逐小时电力负荷、天气和日历数据进行预处理，通过 VMD-DTW 相似性分析辅助选择源域建筑，再构造源域/目标域数据集，最后比较 LSTM 迁移学习策略在目标域小样本场景下的表现。

基本流程：

```text
data_preprocess.ipynb
→ similarity_analysis.ipynb
→ prepare_transfer_data.ipynb
→ transfer_learning.ipynb
→ visualization.ipynb
```

`multistep_comparison.ipynb` 是独立的多步预测对比实验，只验证不同 `HORIZON` 下 `目标域训练` 与 `全层微调` 的多步表现。

## Notebook 职责

| Notebook | 作用 |
| --- | --- |
| `data_preprocess.ipynb` | 处理原始建筑负荷、天气和日历数据，生成各建筑特征文件 |
| `similarity_analysis.ipynb` | 对建筑负荷模式做 VMD-DTW 相似性排序，辅助选择源域/目标域组合 |
| `prepare_transfer_data.ipynb` | 根据 `src/config.py` 中的源域/目标域配置，生成标准化迁移学习数据 |
| `transfer_learning.ipynb` | 训练源域模型，运行源域直测、目标域训练和三种迁移学习策略，并输出单 seed 与多 seed 结果 |
| `multistep_comparison.ipynb` | 对比 `HORIZON = [1, 6, 12, 24]` 下目标域训练和全层微调的多步预测表现 |
| `visualization.ipynb` | 汇总生成负荷曲线、相关性分析和迁移学习结果图 |

## 关键配置

共享配置位于 `src/config.py`：

- `BUILDINGS`：参与实验的建筑列表
- `SOURCE_BUILDING` / `TARGET_BUILDING`：当前迁移学习实验使用的源域和目标域，由人工根据相似性分析结果配置
- `TARGET_SAMPLE_START` / `TARGET_SAMPLE_END`：目标域小样本窗口
- `BASE_DIR` / `BUILDINGS_DIR` / `SCALER_DIR` / `MODEL_DIR` / `FIGURES_DIR`：输出路径
- `TARGET_COL` / `TIME_COL`：负荷目标列和时间列
- `SEED`：主实验随机种子

`src/split_standardize.py` 中的 `TRAIN_RATIO = 0.70` 通过 `src/__init__.py` 暴露为公共 API，供 notebook 对齐数据切分和相似性选择窗口。`VAL_RATIO = 0.15` 仍是 `split_standardize.py` 内部默认参数。

模型超参数如 `LOOKBACK`、`HORIZON`、`BATCH_SIZE`、`EPOCHS`、`HIDDEN_SIZE` 保持在各 notebook 内部定义，便于实验调整。

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

`similarity_analysis.ipynb` 对每个建筑的负荷独立标准化后计算 VMD-DTW，相似性排序以 `加权DTW` 升序为主判据，`相似度(%)` 只作为辅助解释指标。

为避免评估泄漏，相似性分析不使用完整目标小样本窗口，而是在 `TARGET_SAMPLE_START <= timestamp < TARGET_SAMPLE_END` 内取所有建筑公共时间索引的前 `TRAIN_RATIO` 部分作为相似性选择窗口。`determine_optimal_k()` 和成对 VMD-DTW 计算都使用这个窗口。该 notebook 不导入 `TARGET_BUILDING`，因为源域和目标域是在相似性排序后由人工确认。

相似性结果输出：

- `data/相似性分析汇总.csv`
- `data/相似性分析完整结果.json`
- `data/figures/原始负荷对比.png`
- `data/figures/IMF{n}对比.png`

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

- `LSTMPredictor`：单层 LSTM + 线性输出层，`horizon` 控制输出维度。
- `FeatureExtractorRegressor`：复用并冻结预训练 LSTM，接 `Linear(128, 64)` 和 `Linear(64, horizon)` 作为回归头。

公共训练工具在 `src/training.py`：

- `set_seed()`
- `LoadDataset`
- `EarlyStopping`
- `run_epoch()`
- `predict()`
- `evaluate()`：返回 `MAE`、`RMSE`、`CV-RMSE`、`MAPE`、`R2`

序列构造由 `src/preprocessing.py:create_sequences()` 完成，要求时间戳逐小时连续；输出形状为 `X=(N, lookback, feature_count)`，`y=(N, horizon)`。`transfer_learning.ipynb` 的数据流保持为：DataFrame → `create_sequences()` 生成 ndarray → `LoadDataset` 包装为 `torch_datasets`；各策略函数内部按需构造 `DataLoader`。

完整训练循环保留在 notebook 中，不下沉到 `src/`。`src/` 只保留可复用的模型、数据集、指标和单 epoch 训练/验证函数。

`transfer_learning.ipynb` 中多 seed 执行编排直接写在 `[*REPEAT_SEEDS, SEED]` 循环体中；多 seed 汇总和相对 `目标域训练` 的配对统计检验直接写在“实验结果汇总”代码单元中。除 `mean_ci()` 这类会重复使用的小型统计辅助函数外，不为只调用一次的执行编排或汇总逻辑额外定义函数。

`transfer_learning.ipynb` 当前包含的策略：

1. `源域直测（Source-only）`：只在源域训练，不在目标域迁移，直接在目标测试集预测。
2. `目标域训练（Target-only）`：不在源域训练，只在目标域训练集上从零训练模型，作为主参照。
3. `全层微调（Full fine-tune）`：加载源域训练权重后在目标域微调全部层。
4. `冻结LSTM微调（Frozen-LSTM fine-tune）`：冻结 LSTM 层，微调其它层。
5. `冻结特征回归（Frozen-feature regression）`：冻结特征提取层，重新训练新的回归头。

当前训练设置：

- 源域训练与 `目标域训练`：`lr=1e-3`，`weight_decay=1e-4`
- `全层微调`：`lr=1e-4`，`weight_decay=1e-4`
- `冻结LSTM微调` 与 `冻结特征回归`：`lr=1e-3`，`weight_decay=1e-4`
- 各训练函数内部创建 `EarlyStopping(patience=5, restore_best_weights=True)` 并传入 `fit_model()`，便于后续按策略单独调整
- `ReduceLROnPlateau(factor=0.5, patience=4)`
- 多 seed：`REPEAT_SEEDS = [7, 21, 84, 2024]`，实际执行顺序直接使用 `[*REPEAT_SEEDS, SEED]`

每个 seed 运行都会保存模型和图表；由于保存路径固定，后执行的 seed 会覆盖同名模型和图表。`[*REPEAT_SEEDS, SEED]` 将主 `SEED` 放在最后，确保最终保存的模型和图表来自主 `SEED`；CSV 结果仍保留主 `SEED` 单次结果和全部 seed 的统计结果。

## 结果文件

迁移学习输出：

- `models/pretrained_source.pt`
- `models/transfer_target_only.pt`
- `models/transfer_full_fine_tune.pt`
- `models/transfer_frozen_lstm_fine_tune.pt`
- `models/transfer_frozen_feature_regression.pt`
- `data/transfer_learning_results.csv`
- `data/transfer_learning_seed_results.csv`
- `data/transfer_learning_seed_summary.csv`
- `data/transfer_learning_seed_tests.csv`

`transfer_learning_results.csv` 由 `transfer_learning.ipynb` 生成，预期包含 `CV-RMSE`。如果该列缺失，应重跑 `transfer_learning.ipynb`，不要手动补列。

多步预测输出：

- `models/multistep_source_h{1,6,12,24}.pt`
- `data/multistep_step1_comparison.csv`
- `data/multistep_per_step_results.csv`

`multistep_comparison.ipynb` 只保留两种结果策略：`目标域训练（Target-only）` 和 `全层微调（Full fine-tune）`。源域训练只用于为全层微调提供初始权重，不作为结果策略展示。该 notebook 按当前 `HORIZON` 在循环内构造序列和 `torch_datasets`，各训练步骤内部自行构造 `DataLoader`、`EarlyStopping` 和学习率调度器。

主要图表输出位于 `data/figures/`，包括负荷曲线、特征相关性、相似性分析、迁移学习预测对比和多步预测对比图。具体文件以各 notebook 的保存路径为准。

## 维护约定

- 以代码和 notebook 为准维护本文档；本文档不作为 changelog 使用。
- 不记录某次运行的指标数值、最佳 epoch、排序结论或图表解释。
- 修改实验逻辑后，同步更新相关 notebook、`src/` 公共接口说明和本文件中的结构性描述。
