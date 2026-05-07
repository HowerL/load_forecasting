# 训练相关

import os
import random
from contextlib import nullcontext
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def set_seed(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100


def evaluate(y_true_std: np.ndarray, y_pred_std: np.ndarray, scaler) -> dict:
    y_true = scaler.inverse_transform(y_true_std.reshape(-1, 1)).ravel()
    y_pred = scaler.inverse_transform(y_pred_std.reshape(-1, 1)).ravel()

    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': mape(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }


class LoadDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EarlyStopping:
    """
    早停回调
    当验证损失连续patience个epoch没有改善时停止训练
    """
    def __init__(self, patience: int = 8, min_delta: float = 0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.best_loss = float('inf')
        self.best_epoch = -1
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss: float, model, epoch: int = -1) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def restore(self, model):
        # 恢复最佳权重
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)

    def get_best_info(self) -> dict:
        """返回最佳训练信息"""
        return {
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'best_weights': self.best_weights
        }


def run_epoch(model, dataloader, criterion, device, optimizer=None) -> float:
    """
    运行一个epoch的训练或验证

    Args:
        model: PyTorch模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备
        optimizer: 优化器（None时为验证模式）

    Returns:
        平均损失
    """
    model.train() if optimizer else model.eval()
    total_loss = 0

    context = torch.no_grad() if optimizer is None else nullcontext()
    with context:
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            if optimizer:
                optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            if optimizer:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(dataloader.dataset)


def predict(model, dataloader, device) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.no_grad():
        for X_batch, _ in dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.append(outputs.cpu().numpy())
    return np.concatenate(predictions, axis=0).ravel()
