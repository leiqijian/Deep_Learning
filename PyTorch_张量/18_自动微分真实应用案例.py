'''
案例：
    演示自动微分的真实应用场景
结论：
    1. 先前向传播 （正向传播），计算出预测值z
    2. 预测值 和 真实值 相结合，得到损失函数。
    3. 损失函数求导，得到梯度
    4. 结合权重更新公式 ： w新 = w旧 - 学习率 * 梯度。得到新的权重
    5. 重复1~4的动作。不断逼近损失最小的权重值
'''

import torch


# 1. 定义特征x， 表示输入值 假设：2行5列，全1矩阵
x = torch.ones(2, 5)
print(x)

# 2. 定义y，表示真实值标签。假设：2行3列，全0矩阵
y = torch.zeros(2, 3)
print(y)

# 3. 初始化 权重 和 偏置  （初学案例省略了转置步骤，先跑通流程）
# 因为要得到一个（2， 3）的预测值，结合矩阵乘法公式，特征x需要乘一个（5， 3）的矩阵才能得到一个（2， 3）的预测矩阵
w = torch.randn(5, 3, requires_grad=True)
print(f'W_old , {w}')

# 因为标签值是一个（2， 3）的矩阵，所以对应也要有3个偏置 y = w1x1 + w2x2 + ...+ b 每列标签值都有一个b
b = torch.randn(3, requires_grad=True)


# 4. 前向传播，计算出预测值z
z = torch.matmul(x, w) + b
# z = x @ w + b
print(f'Z_pred : {z}')

# 5. 定义损失函数
criterion = torch.nn.MSELoss() #torch框架中集成了很多的模型和损失函数，这里采用了均方误差作为损失函数公式
loss = criterion(z, y) #把预测值和真实值放到损失函数中

# 6. 对损失函数进行自动微分，求导，得到新的梯度
loss.sum().backward()

# 7. 打印w b 结合权重更新公式，得到新的权重值
print(f'W_new , {w.grad}')
print(f'B_new , {b.grad}')
