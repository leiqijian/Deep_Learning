import torch
import numpy as np

# numpy数组转换成张量
# torch.from_numpy(ndarray): 共享内存, 对ndarray数组进行copy()
# torch.tensor(data=ndarray): 不共享内存
def dm02():
	n1 = np.array([[1, 2, 3], [4, 5, 6]])
	# 转换成张量
	# 共享内存
	t1 = torch.from_numpy(n1)
	# 不共享内存
	# t1 = torch.from_numpy(n1.copy())
	# t1 = torch.tensor(data=n1)
	print('t1->', t1)
	print('t1类型->', type(t1))
	# 修改张量元素
	t1[0][0] = 8888
	print('t1修改后->', t1)
	print('n1->', n1)