import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# 可视化工具


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
        y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(y_true.shape)
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

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


def _record_order(records: list, key: str) -> list:
    """按记录出现顺序返回去重后的分组名称。"""
    ordered = []
    for record in records:
        value = record[key]
        if value not in ordered:
            ordered.append(value)
    return ordered


def plot_multiseed_prediction_curves(records: list, output_dir, strategies: list = None, show: bool = False) -> None:
    """按策略分别保存多 seed 预测结果对比曲线。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    strategy_order = strategies if strategies is not None else _record_order(records, '策略')

    for strategy in strategy_order:
        strategy_records = [record for record in records if record['策略'] == strategy]
        if not strategy_records:
            continue

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(
            strategy_records[0]['y_true'],
            label='真实值',
            color='black',
            linewidth=1.8,
        )
        for idx, record in enumerate(strategy_records):
            ax.plot(
                record['y_pred'],
                label='预测值' if idx == 0 else '_nolegend_',
                color='tab:orange',
                linewidth=1.1,
                alpha=0.55,
            )

        ax.set_title(strategy)
        ax.set_xlabel('Sample')
        ax.set_ylabel('kWh')
        ax.grid(True, alpha=0.3)
        ax.legend(facecolor='white', edgecolor='black', labelcolor='black')
        fig.tight_layout()
        fig.savefig(output_dir / f'{strategy}_预测对比.png', dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)


def plot_multiseed_loss_curves(records: list, output_dir, stages: list = None, show: bool = False) -> None:
    """按训练阶段分别保存多 seed 损失曲线。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_order = stages if stages is not None else _record_order(records, '阶段')

    for stage in stage_order:
        stage_records = [record for record in records if record['阶段'] == stage]
        if not stage_records:
            continue

        fig, ax = plt.subplots(figsize=(10, 4))
        for idx, record in enumerate(stage_records):
            train_epochs = np.arange(1, len(record['train_losses']) + 1)
            val_epochs = np.arange(1, len(record['val_losses']) + 1)
            ax.plot(
                train_epochs,
                record['train_losses'],
                label='训练损失' if idx == 0 else '_nolegend_',
                color='tab:blue',
                linewidth=1.1,
                alpha=0.55,
            )
            ax.plot(
                val_epochs,
                record['val_losses'],
                label='验证损失' if idx == 0 else '_nolegend_',
                color='tab:orange',
                linewidth=1.1,
                alpha=0.55,
            )

        ax.set_title(stage)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE')
        ax.grid(True, alpha=0.3)
        ax.legend(facecolor='white', edgecolor='black', labelcolor='black')
        fig.tight_layout()
        fig.savefig(output_dir / f'{stage}_损失曲线.png', dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)
