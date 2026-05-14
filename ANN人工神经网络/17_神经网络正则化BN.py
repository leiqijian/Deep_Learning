'''
批量归一化：是正则化的一种，用于解决 缓解模型 过拟合的情况。名字是叫归一化，但实际上是做标准化的处理

思路：
    先对数据做标准化（会丢失一些信息），然后再对数据做 缩放（λ，理解为w权重） 和 平移（β），再找补回一些信息
应用场景：
    计算机视觉处理，图片识别用到多

    nn.BatchNorm1d(): 主要应用于全连接层或者处理一维数据网络，例如文本处理，接收形状为(N, num_features)的张量作为输入
    nn.BatchNorm2d(): 主要应用于卷积神经网络，处理二维图像数据或者特征图，接收形状为(N,C,H,W)的张量作为输入
    nn.BatchNorm3d(): 主要应用于三维卷积神经网络（3D CNN），处理三维数据，例如视觉或者图像，接收形状为(N,C,D,H,W)的张量作为输入
'''

'''
                                    一维数据 vs 二维数据
根本区别在于：
    一个样本是向量（一维）还是矩阵（二维）
    
一维数据 
    -- 一个样本是一条记录，由一组特征值组成 (向量)
    --  3个样本，每个样本4个特征（比如：身高、体重、年龄、收入）
        data_1d = torch.randn(3, 4)
        样本1: [身高, 体重, 年龄, 收入]
        样本2: [身高, 体重, 年龄, 收入]
        样本3: [身高, 体重, 年龄, 收入]
    --  nn.Linear
        做的事情： y = xW^T + b，即把输入向量乘以权重矩阵
    -- linear = nn.Linear(4, 3)   # 4个输入特征 → 3个输出特征
        
二维数据
    -- 一个样本本身就是一个二维矩阵（行×列）
    -- data_2d = torch.randn(1, 3, 3, 4) 1张图片，3个通道（RGB），每通道3行×4列
        R:  [0.1, 0.3, 0.5, 0.2]    ← 第1行
            [0.4, 0.2, 0.8, 0.1]    ← 第2行
            [0.6, 0.3, 0.1, 0.9]    ← 第3行
        R就是一个样本，
    -- nn.Conv2d(in_channels, out_channels, kernel_size)
        做的事情： 用一个小滑窗（如 3×3）在图像上滑动，每个位置做局部加权求和。它保留空间结构：
                 conv = nn.Conv2d(3, 2, kernel_size=3, padding=1)
                 3个输入通道 → 2个输出通道
                 输入 (1, 3, 3, 4) → 输出 (1, 2, 3, 4) ✅
                 空间尺寸不变，依旧是3行×4列作为一个样本（padding=1保证），通道从3变2
    

'''

import torch
import torch.nn as nn
from statsmodels.tsa.vector_ar import output


def demo01():

    # 创建1张图像，3通道，3行4列的像素点，RGB，红绿蓝三色
    input_2d = torch.randn(size=(1, 3, 3, 4))
    print('input_2d->', input_2d)

    l1 = nn.Linear(2, 3)

    conv = nn.Conv2d(3, 2, kernel_size=3, padding=1)
    output = conv(input_2d)
    # 创建BN层, 标准化 ->一定是在激活函数前进行标准化
    # num_features: 输入样本的通道数
    # eps: 小常数, 避免除0
    # momentum: 指数移动加权平均值
    # affine: 默认True, 引入可学习的γ和β参数
    bn2d = nn.BatchNorm2d(num_features=2, eps=1e-5, momentum=0.1, affine=True)
    ouput_bn2d = bn2d(output)
    print('ouput_bn2d->', ouput_bn2d)

def demo02():
    input_2d = torch.randn(size=(1, 2, 3, 4))
    print('input_2d->', input_2d)
    # todo:2-创建BN层, 标准化 ->一定是在激活函数前进行标准化
    # num_features: 输入样本的通道数
    # eps: 小常数, 避免除0
    # momentum: 指数移动加权平均值
    # affine: 默认True, 引入可学习的γ和β参数
    bn2d = nn.BatchNorm2d(num_features=2, eps=1e-5, momentum=0.1, affine=True)
    ouput_2d = bn2d(input_2d)
    print('ouput_2d->', ouput_2d)

if __name__ == '__main__':
    demo01()
    # demo02()