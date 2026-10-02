"""模型：2-2-1 的 ReLU 网络。"""

import torch
from torch import nn


class XORNet(nn.Module):
    """h = ReLU(W1 x + b1), y = W2 h + b2 (no output activation)."""

    def __init__(self):
        super().__init__()
        # nn.Linear(in, out) 就是一层 "W x + b"：它内部自带一个 out×in 的权重矩阵 W
        # 和一个长度为 out 的偏置 b，两者都是随机初始化的，训练时由优化器更新。

        # 第一层（隐藏层）：2 个输入 (x1, x2) -> 2 个隐藏单元 (h1, h2)
        #   self.hidden.weight 对应 README 里的 W^(1)，形状 2×2
        #   self.hidden.bias   对应 README 里的 b^(1)，形状 2
        self.hidden = nn.Linear(2, 2)

        # 第二层（输出层）：2 个隐藏单元 (h1, h2) -> 1 个输出 y
        #   self.output.weight 对应 README 里的 W^(2)，形状 1×2
        #   self.output.bias   对应 README 里的 b^(2)，形状 1
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        # z = W^(1) x + b^(1)      隐藏层的预激活值
        z = self.hidden(x)
        # h = max(0, z)            ReLU：负数变 0，正数不变，把四个点"折叠"成三个
        h = torch.relu(z)
        # y = W^(2) h + b^(2)      输出层不加激活函数，直接输出一个实数
        return self.output(h)
