'''
回归任务常用损失函数
    MAE:
        公式：
            误差绝对值之和 / 样本总数
        类似L1正则化，权重可以降维为0，数据会变得稀疏

        弊端：
            在0点不可导，可能错过最小值

    MSE:

    Smooth L1

'''
import torch
from torch import nn


def demo01():
    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)

    y_pred = torch.tensor([1.0, 1.0, 1.9], dtype=torch.float32, requires_grad=True)

    criterion = nn.L1Loss()
    loss = criterion(y_pred, y_true)
    # (1 + 1 + 0.1) / 3 = 0.7
    print(loss)

if __name__ == '__main__':
    demo01()
