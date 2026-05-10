# 动量法计算梯度实际上计算的是当前时刻的指数移动加权平均梯度值
'''
梯度下降相关介绍:
    概述：
        梯度下降是结合 本次损失函数的导数（作为梯度）基于学习率 来更新权重
    公式：
        w新 = W旧 - α * 梯度
    存在问题：
        1. 可能遇到鞍点，梯度为0，不再更新
        2. 可能遇到平缓区，梯度下降变慢
        3. 遇到局部最小值
    解决思路：
        从梯度下降公式的 学习率 和 梯度 入手进行优化
        1. 动量法： Momentum
        2. 自适应学习率：AdaGrad， RMSProp
        3. 综合衡量：Adam

    动量法Momentum：
        公式：
            St = β * St-1 + （1 - β） * Gt
        解释：
            St：本次的指数移动加权平均结果
            β： 调节权重系数，越大，越依赖历史指数平均值
            St-1： 历史的指数移动加权平均结果
            Gt： 本次计算的梯度

    自适应学习率 AdaGrad：
        公式：
            累计平方梯度：
                St = St-1 + Gt * Gt
                解释：
                    St：  累计平方梯度
                    St-1：历史累计平方梯度
                    Gt:   本次梯度
            学习率：
                新学习率 = 旧学习率 / (sqrt(St) + 小常数)
                解释：
                    小常数： 1e-10, 目的是防止分母变为0
            梯度下降公式更新为：
                W新 = W旧 - 调整后的学习率 *Gt
        缺点：
            可能会导致学习率过早，过量的降低，导致后期学习率太小，难以找到最优解

    自适应学习率 RMSProp：
        公式：
            指数加权平均 累计 历史 平方梯度：
                St =  β * St-1 + (1 - β)Gt * Gt
                解释：
                    St：  累计平方梯度
                    St-1：历史累计平方梯度
                    Gt:   本次梯度
                    β:    调和权重系数
            学习率：
                新学习率 = 旧学习率 / (sqrt(St) + 小常数)
                解释：
                    小常数： 1e-10, 目的是防止分母变为0
            梯度下降公式更新为：
                W新 = W旧 - 调整后的学习率 *Gt
        优点：
            通过引入 β ，控制历史梯度 对历史信息获取的多与少


'''

import torch
from jedi.inference import param
from torch import optim


def dm01():
    # todo: 1-初始化权重参数
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    # 自定义损失函数, 实际工作中调用不同任务的损失函数, 交叉熵损失/MSE损失...
    loss = ((w ** 2) / 2.0).sum()
    # todo: 2-创建优化器函数对象 SGD->动量法
    # rmsprop : alpha调和权重系数
    optimizer = optim.rmsprop(param = [w], lr=0.01, alpha=0.9)
    # todo: 3-计算梯度值
    optimizer.zero_grad() # 清空上一轮梯度
    loss.sum().backward() # 反向传播更新梯度
    # todo: 4-更新权重参数 梯度更新
    optimizer.step()  # 用梯度更新参数
    print('w.grad->', w.grad)
    # 第二次计算
    loss = ((w ** 2) / 2.0).sum()
    optimizer.zero_grad()    # 1. 清空上轮梯度
    loss.backward()          # 2. 反向传播计算新梯度
    optimizer.step()         # 3. 用梯度更新参数
    print('w.grad->', w.grad)


if __name__ == '__main__':
    dm01()