import torch
import numpy as np


# 张量转换成numpy数组
# tensor.numpy(): 共享内存, 修改一个另外一个也跟着变, 可以通过copy()函数不共享内存
def dm01():
    t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print('t1->', t1)
    # 转换成numpy数组
    # n1 = t1.numpy()
    n1 =     t1.numpy().copy()
    print('n1->', n1)
    print('n1的类型->', type(n1))
    # 修改n1的第一个值
    # [0][0]->第一行第一列的元素
    n1[0][0] = 100
    print('n1修改后->', n1)
    print('t1->', t1)

if __name__ == '__main__':
    dm01()
    dm01()
    dm01()