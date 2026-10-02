"""生成数据：XOR 的四个样本。"""

import torch


def make_xor_data():
    """返回 (X, Y)：X 是 4×2 的输入，Y 是 4×1 的标签。"""
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    return X, Y
