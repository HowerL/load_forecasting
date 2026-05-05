# 数据分割与标准化

from pathlib import Path
from typing import Dict
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from .config import TARGET_COL

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def standardize_load(
    df: pd.DataFrame,
    load_scaler: StandardScaler = None,
    fit_scaler: bool = True,
):
    df = df.copy()
    mask = df[TARGET_COL].notna()

    if fit_scaler:
        load_scaler = StandardScaler()
        load_scaler.fit(df.loc[mask, [TARGET_COL]])

    df.loc[mask, TARGET_COL] = load_scaler.transform(
        df.loc[mask, [TARGET_COL]]
    ).ravel()

    return df, load_scaler


def prepare_source_domain(
    df: pd.DataFrame,
    scaler_dir: Path,
) -> Dict:
    """
    准备源域数据（全量数据标准化，不分割）
    仅标准化负荷（天气已在预处理阶段标准化）。
    """
    df_std, load_scaler = standardize_load(df, fit_scaler=True)

    scaler_dir = Path(scaler_dir)
    scaler_dir.mkdir(exist_ok=True, parents=True)
    joblib.dump(load_scaler, scaler_dir / 'source_load_scaler.joblib')

    return {
        'df': df_std,
        'load_scaler': load_scaler,
    }


def prepare_target_domain(
    df: pd.DataFrame,
    sample_start: str,
    sample_end: str,
    scaler_dir: Path,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> Dict:
    """
    准备目标域数据（选取小样本时间段后分割）
    仅标准化负荷（天气已在预处理阶段标准化）。
    """
    time_col = 'timestamp'
    mask = (df[time_col] >= sample_start) & (df[time_col] < sample_end)
    df_sample = df[mask].reset_index(drop=True)

    n = len(df_sample)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df_sample.iloc[:train_end].copy()
    val_df = df_sample.iloc[train_end:val_end].copy()
    test_df = df_sample.iloc[val_end:].copy()

    train_df, load_scaler = standardize_load(train_df, fit_scaler=True)
    val_df, _ = standardize_load(val_df, load_scaler=load_scaler, fit_scaler=False)
    test_df, _ = standardize_load(test_df, load_scaler=load_scaler, fit_scaler=False)

    scaler_dir = Path(scaler_dir)
    scaler_dir.mkdir(exist_ok=True, parents=True)
    joblib.dump(load_scaler, scaler_dir / 'target_load_scaler.joblib')

    return {
        'train_df': train_df.reset_index(drop=True),
        'val_df': val_df.reset_index(drop=True),
        'test_df': test_df.reset_index(drop=True),
        'load_scaler': load_scaler,
    }