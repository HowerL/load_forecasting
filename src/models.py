# PyTorch LSTM 模型定义

import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, use_activation: bool = False):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.use_activation = use_activation
        self.activation = nn.LeakyReLU(0.1) if use_activation else None
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 取最后一个时间步的输出
        out = lstm_out[:, -1, :]
        if self.use_activation:
            out = self.activation(out)
        out = self.fc(out)
        return out


class FeatureExtractorRegressor(nn.Module):
    """
    特征提取器 + 新回归层（迁移学习策略3）

    将预训练的LSTM作为固定特征提取器，训练新的回归层。
    """
    def __init__(self, pretrained_model: LSTMPredictor, feature_dim: int = 128, use_activation: bool = False):
        super(FeatureExtractorRegressor, self).__init__()

        # 加载预训练的LSTM
        self.lstm = nn.LSTM(
            pretrained_model.lstm.input_size,
            pretrained_model.lstm.hidden_size,
            batch_first=True
        )

        # 加载预训练权重
        self.lstm.load_state_dict(pretrained_model.lstm.state_dict())

        # 冻结LSTM
        for param in self.lstm.parameters():
            param.requires_grad = False

        # 新的回归层
        self.use_activation = use_activation
        self.activation = nn.LeakyReLU(0.1) if use_activation else None
        self.fc1 = nn.Linear(feature_dim, feature_dim // 2)
        self.fc2 = nn.Linear(feature_dim // 2, 1)

    def forward(self, x):
        # 使用冻结的LSTM提取特征
        lstm_out, _ = self.lstm(x)
        features = lstm_out[:, -1, :]

        # 新的回归层
        out = self.fc1(features)
        if self.use_activation:
            out = self.activation(out)
        out = self.fc2(out)
        return out
