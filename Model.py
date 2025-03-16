import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F


class ProteinInteractionModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(ProteinInteractionModel, self).__init__()

        # 投影模块
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 残基接触模块
        self.contact_module = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )

        # 对称化卷积操作
        self.symmetric_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        # 接触图生成卷积
        self.contact_map_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)  # 1x1 卷积压缩 hidden_dim 到 1

        # 相互作用预测模块
        self.interaction_prediction = nn.Linear((242 // 2) * (242 // 2), output_dim)

    def forward(self, E1, E2):
        # 投影模块
        Z1 = self.projection(E1)  # (batch_size, seq0, hidden_dim)
        Z2 = self.projection(E2)  # (batch_size, seq1, hidden_dim)

        # 残基接触模块
        diff = Z1.unsqueeze(2) - Z2.unsqueeze(1)  # (batch_size, seq0, seq1, hidden_dim)
        mul = Z1.unsqueeze(2) * Z2.unsqueeze(1)  # (batch_size, seq0, seq1, hidden_dim)
        combined = torch.cat([diff, mul], dim=-1)  # (batch_size, seq0, seq1, hidden_dim * 2)

        # 通过全连接层处理
        B = self.contact_module(combined.view(-1, combined.size(-1)))  # (batch_size * seq0 * seq1, hidden_dim)
        B = B.view(combined.size(0), combined.size(1), combined.size(2), -1)  # (batch_size, seq0, seq1, hidden_dim)

        # 对称化操作前填充0
        max_len = max(B.size(1), B.size(2))  # 取seq0和seq1的最大长度
        pad1 = max_len - B.size(1)  # 需要填充的seq0长度
        pad2 = max_len - B.size(2)  # 需要填充的seq1长度
        B = F.pad(B, (0, 0, 0, pad2, 0, pad1))  # 在seq0和seq1维度上填充0

        # 对称化卷积操作
        B = B.permute(0, 3, 1, 2)  # (batch_size, hidden_dim, seq0_padded, seq1_padded)
        B_sym = (B + B.permute(0, 1, 3, 2)) / 2  # 对称化操作
        C = self.symmetric_conv(B_sym)  # (batch_size, hidden_dim, seq0_padded, seq1_padded)
        C = torch.sigmoid(C)

        # 使用 1x1 卷积生成接触图
        contact_map = self.contact_map_conv(C).squeeze(1)  # (batch_size, seq0_padded, seq1_padded)

        # 对接触图进行稀疏化
        threshold = contact_map.mean() + 0.5 * contact_map.std()  # 可调整系数
        contact_map_sparse = F.relu(contact_map - threshold)

        # 对接触图进行最大池化
        contact_map_pooled = F.max_pool2d(contact_map_sparse.unsqueeze(1), kernel_size=2,
                                          stride=2)  # (batch_size, 1, seq0_padded//2, seq1_padded//2)
        contact_map_pooled = contact_map_pooled.squeeze(1)  # (batch_size, seq0_padded//2, seq1_padded//2)

        # 展平接触图
        Q_flat = contact_map_pooled.reshape(contact_map_pooled.size(0),
                                            -1)  # (batch_size, (seq0_padded//2) * (seq1_padded//2))

        # 直接通过全连接层输出概率
        interaction_prob = torch.sigmoid(self.interaction_prediction(Q_flat))  # (batch_size, output_dim)

        return interaction_prob,contact_map