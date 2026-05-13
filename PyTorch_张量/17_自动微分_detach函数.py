'''
回顾：
    自动微分 = 求导，即：基于损失函数，计算梯度
    结合权重更新公式；w新 = w旧 - 学习率 * 梯度

问题：
    一个张量一旦设置了自动微分，这个张量就不能直接转成numpy的ndarray对象了，需要通过detach()函数复制一份再处理
'''

import torch

# 1. 定义张量
# 参1：数据，参2：是否允许自动微分，参3：设置数据类型为浮点型
t1 = torch.tensor([10, 20], requires_grad=True, dtype=torch.float)

# 2. 尝试把t1转成numpy对象
# n1 = t1.numpy() #RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
# print(n1)

# 3. 通过detach() 函数。拷贝一份张量，然后转换
t2 = t1.detach()
print(t2)

# 4. 测试拷贝后的是否共享同一块空间 -> 共享
t1.data[0] = 100
print(t1)
print(t2)

# 5. 查看两者谁可以自动微分
print(t1.requires_grad) #返回一个tensor，1 -> TRUE
print(t2.requires_grad) #返回一个tensor，0 -> False

# 6. 把t2转成numpy对象
n1 = t2.numpy()
print(n1)