
import torch
from torch import nn

# 超参数
EMBED_DIM = 32  # 嵌入维度
WEIGHT_MIN, WEIGHT_MAX = 0, 10  # 重量范围


class BoxEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.col_embed = nn.Embedding(6, EMBED_DIM)  # 列索引嵌入（6列）
        self.row_embed = nn.Embedding(6, EMBED_DIM)  # 行位置嵌入（6层）
        self.weight_embed = nn.Linear(1, EMBED_DIM)  # 重量线性嵌入
        self.port_embed = nn.Linear(6, EMBED_DIM)  # 卸货港口嵌入
    
    def forward(self, col, row, weight, port):
        # col: [B], row: [B], weight: [B, 1], port: [B, 6]
        return self.col_embed(col) + self.row_embed(row) + self.weight_embed(weight) + self.port_embed(port)