# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **power load forecasting** project using LSTM neural networks. The project processes electricity load data with weather and calendar features to predict future power consumption for a single building.

## Data Pipeline

The project follows this workflow:
1. **Data Preprocessing**: `data_preprocess.ipynb` - Load & weather cleaning, time feature engineering
2. **Split & Standardize**: `data_split_standardize.ipynb` - Train/val split, standardization, saves scalers
3. **Model Training & Evaluation**: `lstm.ipynb`

## Key Architecture

Main implementation in `lstm.ipynb`:

- **Model**: LSTM(128, selu) + Dense(1)
- **Lookback**: 24 hours of historical data
- **Horizon**: 1 hour ahead prediction
- **Data**: Sliding window with 1-hour continuity check
- **Loss**: MAE (L1Loss)
- **Evaluation**: MAE, RMSE, MAPE, R² on original kWh scale
- **Fine-tuning**: Optional validation set fine-tuning (70% tune, 30% test)
- **Visualization**: Training loss curves, prediction comparison plots
- **Callbacks**: EarlyStopping(patience=5), ReduceLROnPlateau
- **Regularization**: weight_decay=1e-3 (Adam optimizer)

## Key Configuration

In `lstm.ipynb`:
- `SEED = 42` - Random seed for reproducibility
- `LOOKBACK = 24` - Input window length (hours)
- `HORIZON = 1` - Prediction horizon (hours)
- `BATCH_SIZE = 72`
- `EPOCHS = 50` - Maximum training epochs (actual: ~24 for base, ~13 for fine-tuning)
- `hidden_size = 128`
- Target column: `hourly_kwh_clean`

## Core Functions

### lstm.ipynb
- `set_seed()` - Set random seeds for reproducibility
- `load_data()` - Load and parse CSV with timestamp
- `create_sequences()` - Build sliding window sequences with continuity check
- `run_epoch()` - Unified train/validate function (optimizer=None for validation)
- `predict()` - Model inference
- `evaluate()` - Calculate MAE, RMSE, MAPE, R²
- `mape()` - MAPE calculation with zero-division protection
- `EarlyStopping` - Early stopping with best weights restoration
- `plot_training_history()` - Plot training/validation loss curves
- `plot_predictions()` - Plot prediction comparison

### data_preprocess.ipynb
- `build_hr_load_series()` - Convert cumulative energy to hourly load
- `fill_missing()` - Multi-level fallback: 7-day hourly mean → 14-day hourly mean → interpolation
- `detect_outliers()` - MAD (Median Absolute Deviation) based outlier detection
- `process_precipitation()` - Create precipitation boolean + log transform features
- `encode_cyclical_feature()` - Sin/cos encoding for cyclical features (hour, day_of_week, month)

### data_split_standardize.ipynb
- Split by month (1-9 train, 10-12 validation)
- Standardize continuous variables using training set statistics
- Save scalers for inverse transformation

## Data Files

**Input:**
- `data/负荷.CSV` - Raw power load data (cumulative kWh)
- `data/天气.CSV` - Raw weather data
- `data/2024日历.csv` - Calendar features (day_of_week, month, is_holiday)

**Processed:**
- `data/负荷_预处理后.csv` - Cleaned power load data
- `data/天气_预处理后.csv` - Cleaned weather data (with one-hot encoding)
- `data/负荷_天气_时间特征.csv` - Merged load, weather, and time features
- `data/train_std.csv` - Training data (standardized, months 1-9)
- `data/val_std.csv` - Validation data (standardized, months 10-12)

**Scalers (saved by data_split_standardize.ipynb):**
- `data/scalers/load_scaler.joblib` - Load target scaler
- `data/scalers/weather_scaler.joblib` - Weather features scaler

**Model:**
- `models/lstm_single_step.pt` - Trained PyTorch model

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
- Jupyter notebooks
- Virtual environment: `.venv/`