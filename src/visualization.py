# 可视化工具

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_training_history(train_losses: list, val_losses: list, title: str = 'Training History'):
    """绘制训练和验证损失曲线"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.grid(True, alpha=0.3)
    ax.plot(train_losses, label='Train MAE', color='tab:blue')
    ax.plot(val_losses, label='Val MAE', color='tab:orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
    ax.set_title(title)
    ax.legend(facecolor='white', edgecolor='black', labelcolor='black')
    plt.tight_layout()
    plt.show()


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str, scaler=None):
    """绘制预测结果对比图"""
    if scaler is not None:
        y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.grid(True, alpha=0.3)
    ax.plot(y_true, label='Real', color='tab:blue')
    ax.plot(y_pred, label='Predict', color='tab:orange')
    ax.set_xlabel('Sample')
    ax.set_ylabel('kWh')
    ax.set_title(title)
    ax.legend(facecolor='white', edgecolor='black', labelcolor='black')
    plt.tight_layout()
    plt.show()


def plot_load_comparison(
    df,
    time_col: str,
    load_col: str,
    title: str,
    color: str = 'steelblue',
    figsize: tuple = (14, 5),
    save_path: str = None
):
    """绘制负荷曲线"""
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(df[time_col], df[load_col], color=color, linewidth=0.8, alpha=0.8)
    ax.set_xlabel('时间')
    ax.set_ylabel('负荷 (kWh)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_dual_axis_comparison(
    df,
    time_col: str,
    col1: str,
    col2: str,
    title: str,
    color1: str = 'steelblue',
    color2: str = 'coral',
    ylabel1: str = '累计电量 (kWh)',
    ylabel2: str = '小时电量 (kWh)',
    save_path: str = None
):
    """绘制双Y轴对比图"""
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # 左Y轴
    ax1.plot(df[time_col], df[col1], color=color1, linewidth=1, label=col1)
    ax1.set_xlabel('时间')
    ax1.set_ylabel(ylabel1, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    # 右Y轴
    ax2 = ax1.twinx()
    ax2.plot(df[time_col], df[col2], color=color2, linewidth=0.5, alpha=0.6, label=col2)
    ax2.set_ylabel(ylabel2, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    ax1.set_title(title)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_monthly_load(df, time_col: str, load_col: str, title_prefix: str = ''):
    """按月绘制负荷曲线"""
    df = df.copy()
    df['month'] = df[time_col].dt.month
    months = sorted(df['month'].unique())
    n_months = len(months)

    fig, axes = plt.subplots(n_months, 1, figsize=(14, 3 * n_months), sharex=False)

    if n_months == 1:
        axes = [axes]

    for i, month in enumerate(months):
        month_data = df[df['month'] == month]
        axes[i].plot(month_data[time_col], month_data[load_col], color='coral', linewidth=0.8, alpha=0.8)
        axes[i].set_title(f'{title_prefix}{month}月 负荷')
        axes[i].set_ylabel('kWh')
        axes[i].grid(True, alpha=0.3)

        axes[i].xaxis.set_major_locator(mdates.DayLocator(interval=5))
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
