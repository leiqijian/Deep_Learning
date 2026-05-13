import torch


# 矩阵乘法: (n, m) * (m, p) = (n, p)  第一个矩阵的行和第二个矩阵的列相乘  @  torch.matmul(input=, ohter=)
def dm01():
	# (2, 2)
	t1 = torch.tensor(data=[[1, 2],
							[3, 4]])
	# (2, 3)
	t2 = torch.tensor(data=[[5, 6, 7],
							[8, 9, 10]])

	# @
	t3 = t1 @ t2
	print('t3->', t3)
	# torch.matmul(): 不同形状, 只要后边维度符合矩阵乘法规则即可
	t4 = torch.matmul(input=t1, other=t2)
	print('t4->', t4)

	# dot()函数
	t5 = torch.tensor([1, 2, 3])
	t6 = torch.tensor([4, 5, 6])
	t7 = t5.dot(t6)
	print('t7->', t7) #tensor(32)
	# 32 这里是因为pytorch有__format__方法，对字符串进行了提取，便于文本拼接，但是t7的类型仍然是tensor
	#不同于 t7.item().返回值类型是int
	print(f't7-> {t7}')


if __name__ == '__main__':
	dm01()