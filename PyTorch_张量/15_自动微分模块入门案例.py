'''
回顾；
    权重更新公式：
        w_new = w_old - α * grad
        grad = 损失函数的导数 (梯度)
    pytorch内置 自动微分模块，损失函数调用backward

细节：
    只有标量张量才可以求导，且大多底层操作都是浮点型需要先转型
'''

import torch

# 1. 定义变量，记录初始权重w，并设置自动微分

w = torch.tensor(10, requires_grad=True, dtype=torch.float)

# 2. 定义loss变量，表示损失函数
loss = 2 * w ** 2

# 3. 计算梯度 梯度为损失函数求导，在这一步对损失函数进行了求导
loss.sum().backward()

# 4. 代入公式 更新梯度
w.data = w.data - 0.01 * w.grad

print(w.data)