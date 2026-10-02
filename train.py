"""训练：只负责更新参数，不知道数据从哪来、也不知道怎么画图。"""

import torch
from torch import nn


def train(model, X, Y, lr=0.05, max_steps=2000, tol=1e-4, on_step=None):
    """用 Adam 做 full-batch 训练，直到 MSE loss 低于 tol 或达到 max_steps。

    on_step(step, loss) 会在初始化后（step 0）和每一次参数更新后被调用一次，
    调用方可以用它来画图、打日志等。返回 (更新次数, 最终 loss)。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    step = 0
    loss = loss_fn(model(X), Y)
    if on_step is not None:
        on_step(step, loss.item())

    while step < max_steps and loss.item() > tol:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
        loss = loss_fn(model(X), Y)
        if on_step is not None:
            on_step(step, loss.item())

    return step, loss.item()
