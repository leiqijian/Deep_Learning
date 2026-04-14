import torch


# tensor.item(): 提取单个元素张量的数值, 张量可以是标量张量/一维张量/二维张量...只要是单个元素即可
def dm01():
	# 数值转换成张量
	# 标量
	t1 = torch.tensor(data=10)  #10
	# 一维
	t2 = torch.tensor(data=[20]) #10
	# 二维
	t3 = torch.tensor(data=[[30]]) #10
	print('t1->', t1) #tensor(10)
	print('t1形状->', t1.shape)
	# 单个元素张量转换成数值, 提取数值
	print('t1.item()->', t1.item()) #10
	print('t2.item()->', t2.item()) #20
	print('t3.item()->', t3.item()) #30


if __name__ == '__main__':
	dm01()