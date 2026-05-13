'''
pytorch 创建线性和随机张量

涉及到的函数：
    torch.arange() 和 torch.linspace() 创建线性张量
    torch.random.initial_seed() 和 torch.random.manual_seed() 随机种子设置
    torch.rand/randn() 创建随机浮点类型张量
    torch.randint(low, high, size=()) 创建随机整数类型张量

'''

import torch
import numpy as np

def dm01():
	t1 = torch.arange(start=0, end=10, step=2)
	print('t1的值是->', t1)
	print('t1类型是->', type(t1))
	t2 = torch.linspace(start=0, end=9, steps=9)
	print('t2的值是->', t2)
	print('t2类型是->', type(t2))

def dm02():
	# (5, 4): 5行4列
	t1 = torch.rand(size=(2, 3))
	print('t1的值是->', t1)
	print('t1类型->', type(t1))
	print('t1元素类型->', t1.dtype)
	print('t1随机种子数->', torch.initial_seed())
	# 设置随机种子数
	# torch.manual_seed(seed=66)
	t2 = torch.randint(low=0, high=10, size=(2, 3))
	print('t2的值是->', t2)
	print('t2类型->', type(t2))
	print('t2元素类型->', t2.dtype)
	print('t2随机种子数->', torch.initial_seed())


if __name__ == '__main__':
    dm01()
    print("----------------------------")
    dm02()