# 可视化工具

import numpy as np
import matplotlib.pyplot as plt


def setup_plot_style(dpi: int = 150):
    """配置 matplotlib 中文字体和全局样式"""
    plt.style.use('default')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = dpi


def plot_training_history(train_losses: list, val_losses: list, title: str = 'Training History', save_path: str = None):
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
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str, scaler=None, save_path: str = None):
    """绘制预测结果对比图"""
    if scaler is not None:
        y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.grid(True, alpha=0.3)
    ax.plot(y_true, label='Real', color='tab:blue')
    ax.plot(y_pred, label='Predict', color='tab:orange')
    ax.set_xlabel('Sample')
    ax.set_ylabel('kWh' if scaler is not None else '标准差')
    ax.set_title(title)
    ax.legend(facecolor='white', edgecolor='black', labelcolor='black')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()