# PyTorch LSTM 模型定义

import torch
import torch.nn as nn
from typing import Optional


class LSTMPredictor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        horizon: int = 1,
    ):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.activation = nn.LeakyReLU(0.1)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 取最后一个时间步的输出
        out = lstm_out[:, -1, :]
        out = self.activation(out)
        out = self.fc(out)
        return out


class FeatureExtractorRegressor(nn.Module):
    """
    固定特征回归策略使用的回归模型。

    将预训练 LSTM 作为固定时序特征提取器，训练新的回归层。
    """
    def __init__(
        self,
        pretrained_model: LSTMPredictor,
        feature_dim: Optional[int] = None,
        horizon: int = 1,
    ):
        super(FeatureExtractorRegressor, self).__init__()

        # 直接引用预训练的LSTM
        self.lstm = pretrained_model.lstm
        if feature_dim is None:
            feature_dim = self.lstm.hidden_size

        # 固定预训练 LSTM 参数
        for param in self.lstm.parameters():
            param.requires_grad = False

        # 新的回归层
        self.activation = nn.LeakyReLU(0.1)
        self.fc1 = nn.Linear(feature_dim, feature_dim // 2)
        self.fc2 = nn.Linear(feature_dim // 2, horizon)

    def forward(self, x):
        # 使用固定的 LSTM 提取特征
        lstm_out, _ = self.lstm(x)
        features = lstm_out[:, -1, :]

        # 新的回归层
        out = self.fc1(features)
        out = self.activation(out)
        out = self.fc2(out)
        return out
