'''
二分类任务的损失函数（BCELoss）:
    公式：
        Loss = -ylog(预测值) - (1 - y)log(1 - 预测值)
        y: 样本x属于某一个类别的真实概率，是0或者1


'''
import torch
import torch.nn as nn

def dm01():
	# 手动创建样本的真实y值
	# y_true =
	y_true = torch.tensor(data=[0, 1, 0], dtype=torch.float32)

	# 手动创建样本的预测y值 -> 模型预测值f(X)
    # 相当于第一个样本0这个分类预测概率为0.6901
    # 相当于第二个样本1这个分类预测概率为0.5432
    # 相当于第三个样本0这个分类预测概率为0.2639
	y_pred = torch.tensor(data=[0.6901, 0.5432, 0.2639], requires_grad=True, dtype=torch.float32)

	# 创建多分类交叉熵损失对象
	# reduction:损失值计算的方式, 默认mean 平均损失值
	criterion = nn.BCELoss()

	# 调用损失对象计算损失值
	# 预测y  真实y
	loss = criterion(y_pred, y_true)
	print(type(loss))
	print(type(loss.detach().numpy()))
	print('loss->', loss)


if __name__ == '__main__':
	dm01()