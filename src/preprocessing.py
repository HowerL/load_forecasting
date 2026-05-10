# 数据预处理

import numpy as np
import pandas as pd
from pathlib import Path

from .config import TIME_COL


def build_hr_load_series(
    raw_df: pd.DataFrame,
    ts_col: str,
    cumulative_col: str,
    hr_load_col: str = "hourly_kwh",
) -> pd.DataFrame:
    df = raw_df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values(ts_col).set_index(ts_col)
    df = df.asfreq("1h")
    df[cumulative_col] = pd.to_numeric(df[cumulative_col], errors="coerce").astype(np.float64)

    # 累计电能 -> 每小时负荷
    df[hr_load_col] = df[cumulative_col].diff(1)
    df[hr_load_col] = df[hr_load_col].mask(df[hr_load_col].le(0), np.nan)
    return df


def fill_missing(s: pd.Series) -> pd.Series:
    """
    前向均值法填充缺失值：对每个t，取 t-24, t-48, ... t-24*days 的值求均值

    1. 正常：缺失点用同序列前7天同小时的均值补全
    2. 回退：若可用样本点不足（<2），回退到前14天同小时
    3. 兜底：若可用样本点仍不足则用线性插值+前向填充兜底
    """
    s_filled = s.copy()

    # 多级回退：7天 -> 14天
    for lookback_days in [7, 14]:
        shifted_list = [s_filled.shift(24 * d) for d in range(1, lookback_days + 1)]
        hist_df = pd.concat(shifted_list, axis=1)
        valid_cnt = hist_df.notna().sum(axis=1)
        hist_mean = hist_df.mean(axis=1, skipna=True)

        # 仅当有足够样本(>=2)且当前值缺失时填充
        fill_mask = s_filled.isna() & (valid_cnt >= 2)
        s_filled.loc[fill_mask] = hist_mean.loc[fill_mask]

    # 兜底：时间序列插值 + 前向填充
    s_filled = s_filled.interpolate(method="time", limit_direction="forward").ffill()
    return s_filled


def detect_outliers(
    s: pd.Series,
    window: int = 168,
    threshold: float = 3.0,
) -> pd.Series:
    """
    滑动MAD（中位数绝对偏差）异常检测
    $$ MAD = median\left (\left |X_i - median(X)\right |\right ) $$
    当 $ \left |X_i - median\right | > threshold\times MAD $ 时判定为异常值，使用滑动窗口计算局部中位数和局部MAD。
    """
    s_num = s.copy()

    rolling_median = s_num.rolling(window=window, center=True, min_periods=window).median()
    rolling_mad = s_num.rolling(window=window, center=True, min_periods=window).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )

    abs_dev = (s_num - rolling_median).abs()
    ratio = abs_dev / rolling_mad.replace(0, np.nan)

    outlier_mask = ratio > threshold

    # 当窗口内 MAD=0 且当前点偏离局部中位数时，补判为异常
    zero_mad_fallback = (rolling_mad == 0) & (abs_dev > 0)
    outlier_mask = outlier_mask | zero_mad_fallback

    return outlier_mask.fillna(False)


def encode_cyclical_feature(value: pd.Series, max_val: int) -> tuple[pd.Series, pd.Series]:
    sin_val = np.sin(2 * np.pi * value / max_val)
    cos_val = np.cos(2 * np.pi * value / max_val)
    return sin_val, cos_val


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='utf-8-sig')
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors='coerce')
    df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)
    return df


def create_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    lookback: int = 24,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    构造滑动窗口序列

    Args:
        df: 数据框
        feature_cols: 特征列名列表
        target_col: 目标列名
        lookback: 输入窗口长度
        horizon: 预测步长

    Returns:
        (X, y) 数组，X形状为 (样本数, lookback, 特征数)，y形状为 (样本数, horizon)
    """
    values = df[feature_cols].astype(np.float32).to_numpy()
    target_idx = feature_cols.index(target_col)

    x, y = [], []
    max_i = len(df) - lookback - horizon + 1
    for i in range(max_i):
        ts_seg = df[TIME_COL].iloc[i:i + lookback + horizon]
        if ts_seg.isna().any():
            continue  # 如有缺失值跳过
        if not (ts_seg.diff().dropna() == pd.Timedelta(hours=1)).all():
            continue  # 如有间断点跳过

        x_i = values[i:i + lookback]
        y_i = values[i + lookback: i + lookback + horizon, target_idx]
        if np.isnan(x_i).any() or np.any(np.isnan(y_i)):
            continue  # 如有缺失值跳过

        x.append(x_i)
        y.append(y_i)

    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)
