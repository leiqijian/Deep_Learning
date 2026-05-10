'''
学习率衰减策略：
    目的：
        相比AdaGrad，RMSProp，Adam等方式，可以通过  等间隔， 指定间隔， 指数等方式，手动控制学习率下降策略
    分类：
        等间隔
        指定间隔
        指数衰减

等间隔学习率衰减：
    optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.5)
        optimizer: 优化器对象
        step_size：间隔轮数。即多少轮调整一次学习率
        gamma：学习率衰减系数，一次衰减多少，lr新 = lr旧 * gamma

指定间隔学习率衰减：

'''# 等间隔: 指定训练次数后修改学习率  lr=lr*gamma
import torch
from torch import optim
import matplotlib.pyplot as plt

def dm01():
    # todo: 1-初始化参数
    # lr epoch iteration
    lr = 0.1
    epoch = 200
    iteration = 10
    # todo: 2-创建数据集
    # y_true x w
    y_true = torch.tensor([0])
    x = torch.tensor([1.0], dtype=torch.float32)
    print(y_true)
    print(x)
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    # todo: 3-创建优化器对象 动量法
    optimizer = optim.SGD(params=[w], lr=lr, momentum=0.9)
    # todo: 4-创建等间隔学习率衰减对象
    # optimizer: 优化器对象
    # step_size: 间隔, 指定训练轮数后修改学习率
    # gamma: 衰减系数 默认0.1
    scheduer = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.5)
    # todo: 5-创建两个列表, 收集训练次数, 收集每次训练lr
    lr_list, epoch_list = [], []
    # todo: 6-循环遍历训练次数
    for i in range(epoch):
        # todo: 7-获取每次训练的次数和lr保存到列表中
        # scheduer.get_last_lr(): 获取最后lr
        lr_list.append(scheduer.get_last_lr())
        epoch_list.append(i)
        # todo: 8-循环遍历, batch计算, 模拟批次
        for batch in range(iteration):
            # 先算预测y值 wx, 计算损失值 (wx-y_true)**2
            y_pred = w * x
            loss = (y_pred - y_true) ** 2
            # 梯度清零
            optimizer.zero_grad()
            # 梯度计算
            loss.backward()
            # 参数更新
            optimizer.step()
        # todo: 9-执行完所有批次，即执行完一轮之后，更新一次学习率，当达到指定轮数的时候，学习率会下降
        scheduer.step()
    print('lr_list->', lr_list)

    plt.plot(epoch_list, lr_list)
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    dm01()