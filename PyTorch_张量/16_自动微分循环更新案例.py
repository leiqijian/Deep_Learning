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

# 1. 定义张量
w = torch.tensor(10, requires_grad=True, dtype=torch.float)

# 2. 定义损失函数，把张量w作为权重参数
loss = w ** 2 + 20

for i in range(1, 101):
    loss = w ** 2 + 20

    if w.grad is not None:
        w.grad.zero_() #因为会自动累加，所以每一次都需要清零

    # 损失函数求导并更新梯度
    loss.sum().backward()


    # 梯度更新 拿到新的权重值
    w.data = w.data - 0.01 * w.grad

    print(f'第{i}次权重{w.grad}, 新梯度{w.data}, loss = {loss}')
