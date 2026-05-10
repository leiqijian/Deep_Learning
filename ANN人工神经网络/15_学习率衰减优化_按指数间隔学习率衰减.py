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
optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[50, 100, 160], gamma=0.5,last_epoch=-1)
    optimizer: 优化器对象
    milestones：指定衰减的 epoch 节点列表，在第 50、100、160 个 epoch时触发学习率衰减
    gamma：学习率衰减系数，一次衰减多少，lr新 = lr旧 * gamma
    last_epoch含义: 上一次训练的 epoch 索引，-1 表示从头开始；恢复训练时可设为上次中断的 epoch 值

按指数间隔学习率衰减：
optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
    optimizer: 优化器对象
    gamma：学习率衰减系数
    与 StepLR/MultiStepLR 不同，ExponentialLR 是每个 epoch都衰减，而不是每隔 N 个 epoch 才衰减一次：

总结：
    - 优先选择指数学习率衰减方法
    - 根据经验设置间隔选择指定间隔学习率衰减方法
    - 简单模型选择等间隔学习率衰减方法

'''
import torch
from torch import optim
import matplotlib.pyplot as plt
# 指数间隔: 前期学习率衰减快, 中期慢, 后期更慢  lr=lr * gamma**epoch
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
	w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
	# todo: 3-创建优化器对象 动量法
	optimizer = optim.SGD(params=[w], lr=lr, momentum=0.9)
	# todo: 4-创建等间隔学习率衰减对象
	# optimizer: 优化器对象
	# gamma: 衰减系数 设置大一些, 初始指数大
	scheduer = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
	# todo: 5-创建两个列表, 收集训练次数, 收集每次训练lr
	lr_list, epoch_list = [], []
	# todo: 6-循环遍历训练次数
	for i in range(epoch):
		# todo: 7-获取每次训练的次数和lr保存到列表中
		# scheduer.get_last_lr(): 获取最后lr
		lr_list.append(scheduer.get_last_lr())
		epoch_list.append(i)
		# todo: 8-循环遍历, batch计算
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
		# todo: 9-更新下一次训练的学习率
		scheduer.step()
	print('lr_list->', lr_list)

	plt.plot(epoch_list, lr_list)
	plt.xlabel("Epoch")
	plt.ylabel("Learning rate")
	plt.legend()
	plt.show()


if __name__ == '__main__':
	dm01()