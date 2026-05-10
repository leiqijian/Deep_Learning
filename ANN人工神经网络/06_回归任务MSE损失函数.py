'''
回归任务常用损失函数
    MAE:
        公式：
            误差绝对值之和 / 样本总数
        类似L1正则化，权重可以降维为0，数据会变得稀疏，下降趋势恒定

        弊端：
            在0点不可导，可能错过最小值

    MSE:
        公式：
            误差平方和 / 样本总数
        类似L2正则化，权重趋近于0，会暴露异常点但同时也会容易梯度爆炸,误差大的话下降快，误差小的话下降慢

    Smooth L1

'''
import torch
from torch import nn


def demo01():
    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)

    y_pred = torch.tensor([1.4, 1.5, 1.9], dtype=torch.float32, requires_grad=True)

    L1 = nn.L1Loss()
    L2 = nn.MSELoss()

    L1_loss = L1(y_pred, y_true)
    L2_loss = L2(y_pred, y_true)
    # (1² + 1² + 0.1²) / 3 = 0.6700
    print(f"L1 Loss: {L1_loss}")
    print(f"L2 Loss: {L2_loss}")

if __name__ == '__main__':
    demo01()
