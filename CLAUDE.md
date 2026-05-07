# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **power load forecasting** project using LSTM neural networks with **transfer learning**. The project processes electricity load data with weather and calendar features to predict future power consumption. It includes a **source domain** (data-rich building for pre-training) and a **target domain** (small-sample building for transfer learning).

我的课题是"基于迁移学习的校园建筑用能负荷预测研究"，这个课题需要依次经过以下处理：数据预处理（异常值检测、缺失值填充、特征工程等）->相似性分析后选取相似性最佳的建筑作为源域->进行数据集分割与标准化->迁移学习。我手上有一些校园建筑逐小时的电力负荷数据集，我需要对其中一栋的指定某段时间作为研究用的"小样本数据"目标域，从其他建筑丰富的数据中选取相似度最高的作为源域来迁移。

## Data Pipeline

The processing flow is:
```
data_preprocess.ipynb → similarity_analysis.ipynb → prepare_transfer_data.ipynb → transfer_learning.ipynb
```

1. **Data Preprocessing**: `data_preprocess.ipynb` - All buildings' load & weather cleaning, time feature engineering
2. **Similarity Analysis**: `similarity_analysis.ipynb` - VMD-DTW similarity analysis to select best source-target pairs
3. **Data Preparation**: `prepare_transfer_data.ipynb` - Prepare source (full data) and target (small sample) domain data
4. **Transfer Learning**: `transfer_learning.ipynb` - Train source model and apply transfer learning strategies

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `data_preprocess.ipynb` | All buildings data preprocessing (generalized for any number of buildings) |
| `similarity_analysis.ipynb` | VMD-DTW similarity analysis to select best source-target pairs |
| `prepare_transfer_data.ipynb` | Split and standardize selected source/target domain data |
| `transfer_learning.ipynb` | Transfer learning experiments with three strategies |
| `visualization.ipynb` | Data visualization summary (load curves, correlation analysis) |

## Key Architecture

### LSTM Model
- **Model**: LSTM(128, selu) + Dense(1)
- **Lookback**: 24 hours of historical data
- **Horizon**: 1 hour ahead prediction
- **Data**: Sliding window with 1-hour continuity check
- **Loss**: MAE (L1Loss)
- **Evaluation**: MAE, RMSE, MAPE, R² on original kWh scale
- **Callbacks**: EarlyStopping(patience=5), ReduceLROnPlateau
- **Regularization**: weight_decay=1e-3 (Adam optimizer)

### VMD-DTW Similarity Analysis
- **VMD**: Variational Mode Decomposition for signal decomposition into IMFs
- **DTW**: Dynamic Time Warping distance for sequence similarity measurement
- **Optimal K Selection**: Based on center frequency stability
- **Weighted Similarity**: Low-frequency IMFs weighted higher (8:4:2:1:1 for K=5)
- **Standardization**: Each building's load data is standardized independently before VMD analysis to focus on pattern similarity rather than magnitude differences

**Core Functions (defined in `similarity_analysis.ipynb`):**
- `determine_optimal_k()` - Determine optimal VMD decomposition level K
- `compute_vmd_dtw_similarity()` - Core computation: VMD + DTW + similarity
- `_compute_pair_similarity_task()` - Parallel task wrapper for ThreadPoolExecutor

### Transfer Learning Strategies
1. **Full Fine-tuning**: Pre-train on source, fine-tune all parameters on target
2. **Partial Fine-tuning**: Freeze LSTM layers, fine-tune FC layers only
3. **Feature Extractor**: Use pre-trained LSTM as fixed feature extractor, train new regressor

## Key Configuration

### Shared Configuration (`src/config.py`)
跨 notebook 共享的配置：
- `BUILDINGS` - 建筑列表
- `SOURCE_BUILDING`, `TARGET_BUILDING` - 源域和目标域建筑
- `TARGET_SAMPLE_START`, `TARGET_SAMPLE_END` - 目标域小样本时间段（用于相似性分析和数据选取）
- `BASE_DIR`, `BUILDINGS_DIR`, `SCALER_DIR`, `MODEL_DIR` - 路径配置
- `TARGET_COL`, `TIME_COL` - 列名配置
- `SEED` - 随机种子

### Notebook-local Hyperparameters
模型超参数（如 `LOOKBACK`, `HORIZON`, `BATCH_SIZE`, `EPOCHS`, `HIDDEN_SIZE`）在各 notebook 中本地定义，便于独立调整。

### Data Split Strategy (`src/split_standardize.py`)

**标准化策略**
- **天气数据**：在 `data_preprocess.ipynb` 阶段使用全量天气数据标准化，所有建筑共享同一份 `weather_scaler.joblib`
- **负荷数据**：每个域独立标准化（建筑间负荷量级差异大）

**源域（数据丰富，用于预训练）**
- 使用全量数据，不分割
- 预训练时从全量数据末尾划分 10% 作为验证集（用于早停）

**目标域（小样本，用于微调与评估）**
- 选取 `TARGET_SAMPLE_START` 到 `TARGET_SAMPLE_END` 时间段的数据
- 按 70%/15%/15% 比例分割为训练/验证/测试集

## src/ Module

Common functions extracted to `src/` module:

### `src/preprocessing.py`
- `build_hr_load_series()` - Convert cumulative energy to hourly load
- `fill_missing()` - Multi-level fallback: 7-day → 14-day hourly mean → interpolation
- `detect_outliers()` - MAD-based outlier detection
- `encode_cyclical_feature()` - Sin/cos encoding for cyclical features
- `load_data()` - Load and parse CSV with timestamp
- `create_sequences()` - Build sliding window sequences with continuity check

### `src/models.py`
- `LSTMPredictor` - LSTM model class
- `FeatureExtractorRegressor` - Feature extractor for transfer learning

### `src/training.py`
- `set_seed()` - Set random seeds for reproducibility
- `mape()` - MAPE calculation with zero-division protection
- `evaluate()` - Calculate MAE, RMSE, MAPE, R²
- `LoadDataset` - PyTorch Dataset
- `EarlyStopping` - Early stopping with best weights restoration
- `run_epoch()` - Unified train/validate function
- `predict()` - Model inference

### `src/split_standardize.py`
- `standardize_load()` - Standardize load feature only (weather already standardized in preprocessing)
- `prepare_source_domain()` - Full data standardization for source domain (no split)
- `prepare_target_domain()` - Small sample selection + split for target domain
- Uses `TARGET_COL` from config (shared column name)

**Note: Internal constants are NOT exported**
- `TRAIN_RATIO` and `VAL_RATIO` are internal constants used only as default arguments, not exported to public API
- To change split ratios, modify the values directly in `split_standardize.py`

### `src/visualization.py`
- `plot_training_history()` - Plot training/validation loss curves
- `plot_predictions()` - Plot prediction comparison

## Data Files

### Input Files
- `data/buildings/{建筑名}.CSV` - Raw power load data (cumulative kWh) for each building
- `data/天气.CSV` - Raw weather data (shared across all buildings)
- `data/2024日历.csv` - Calendar features

### Preprocessed Files
- `data/buildings/{建筑名}_预处理后.csv` - Cleaned load data for each building
- `data/buildings/{建筑名}_特征.csv` - Merged features (load + standardized weather + time) for each building
- `data/天气_预处理后.csv` - Cleaned and standardized weather data (with one-hot encoding)

### Standardized Files (for transfer learning)
- `data/source_train_std.csv` - Source domain full standardized data (for pre-training)
- `data/target_train_std.csv` / `data/target_val_std.csv` / `data/target_test_std.csv` - Target domain standardized data (small sample split)

### Scalers
- `data/scalers/weather_scaler.joblib` - Weather scaler (shared across all buildings, created in preprocessing)
- `data/scalers/source_load_scaler.joblib` - Source domain load scaler
- `data/scalers/target_load_scaler.joblib` - Target domain load scaler

### Model
- `models/pretrained_source.pt` - Pre-trained model on source domain
- `models/transfer_*.pt` - Transfer learning models

### Generated Files
- `data/相似性分析_完整结果.json` - Complete analysis results (K, similarity, DTW distances, center frequencies, weights)
- `data/相似性分析汇总.csv` - Building pair summary (source, target, K, weighted DTW, original DTW, similarity)
- `data/原始负荷对比.png` - Source vs target load comparison
- `data/IMF1对比.png` ~ `data/IMF{n}对比.png` - IMF component comparisons

## Feature Engineering

**Weather features:**
- Continuous: 温度(℃), 风力(级), 风速(km/h), 气压(hPa), 湿度(%), 能见度(km), 云量%
- Precipitation: 是否降水 (boolean), 降水量对数变换 (log1p)
- Categorical: One-hot encoded (天气状况, 风向)

**Time features:**
- Cyclical (sin/cos): hour, day_of_week, month
- Boolean: is_holiday

## Environment

- Python 3.9
- PyTorch
- scikit-learn
- pandas, numpy
- matplotlib
- vmdpy (VMD decomposition)
- dtaidistance (DTW distance)
- Jupyter notebooks
- Virtual environment: `.venv/`