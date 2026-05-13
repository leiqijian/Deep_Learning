'''

'''

import torch

torch.manual_seed(24)

def demo01():
    # 定义2行3列的张量
    t1 = torch.randint(1, 10, size=(2,3))
    #t1.shape[0] 从左往右第一个元素，t1.shape[-1] 从右往左第一个元素
    print(f't1 = {t1}, shape = {t1.shape}, row = {t1.shape[0]}, column = {t1.shape[1]}, {t1.shape[-1]}, {t1.shape[-2]}')
    # 2. 通过reshape() 函数，把t1 -> 3行2列， 1行6列，6行1列
    # t2 = t1.reshape(3, 2)
    # t2 = t1.reshape(1, 6)
    t2 = t1.reshape(6, 1)
    print(f't1 = {t2}, shape = {t2.shape}, row = {t2.shape[0]}, column = {t2.shape[1]}, {t2.shape[-1]}')

def demo02():
    # 定义2行3列的张量
    t1 = torch.randint(1, 10, size=(2,3))
    # 维度索引：dim=0  第0维度 2
    # 维度索引：dim=1  第1维度 3
    # 插入新维度后，最大维度索引会增加，原维度会向后顺移
    print(t1)

    # 在0维上，添加一个维度
    # (1, 2, 3) ,变成了1个2行3列的数据
    t2 = t1.unsqueeze(0)
    print(t2)

    # 在1维上，添加一个维度
    # (2, 1, 3) 变成了2个1行3列的数据
    t3 = t1.unsqueeze(1)
    print(t3)

    # 在2维上，添加一个维度
    # (2, 3, ) 变成了2个3行1列的数据
    t4 = t1.unsqueeze(2)
    print(t4)

    # 删除所有为1的维度
    t5 = torch.randint(1, 10, size=(2, 1, 3, 1, 1))
    print(t5)

    t6 = t5.squeeze()
    print(t6)


def demo03():
    # 定义2行3列的张量
    # dim0 = 2
    # dim1 = 3
    # dim2 = 4
    t1 = torch.randint(1, 10, size=(2, 3, 4))

    # 改变维度 从(2, 3, 4) -> (4, 3, 2)
    t2 = t1.transpose(0, 2) #0维和2维相转换
    print(t2)

    #改变维度 从(2, 3, 4) -> (4, 2, 3)
    t3 = t1.permute(2, 0, 1)
    print(t3)


def demo04():
    t1 = torch.randint(1, 10, size=(2, 3))

    # 判断张量是否连续，即：张量中的顺序 和 内存中的顺序是否一致
    print(t1.is_contiguous())   # true

    # 如果是连续的，可以通过view() 修改上述张量的形状 从(2, 3) -> (3, 2)
    t2 = t1.view(3, 2)
    print(t2)
    print(t2.is_contiguous()) # true

    # 通过transpose()交换之后就不连续，
    t3 = t1.transpose(0, 1)
    print(t3)
    print(t3.is_contiguous()) # false

    # 尝试把t3通过view()， 从(3, 2) -> (2, 3)
    # t4 = t3.view(2, 3) #会报错
    # print(t4)

    # 可以通过contiguous()函数把 t3张量转换为连续张量 ，然后再用view()
    t5 = t3.contiguous().view(2, 3)
    print(t5)



if __name__ == '__main__':
    # demo01()
    # demo02()
    # demo03()
    demo04()