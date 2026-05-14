'''
神经网络正则化作用：
    缓解模型过拟合情况

正则化方式：
    L1，L2正则化
    Dropout正则化：神经元随机失活，让一部分神经元随机死亡，剩下的神经元增加权重
    BN(批量归一化)

'''
import torch
import torch.nn as nn

def demo01():
    # 创建输入层
    input = torch.randint(0, 10, size=(1,4)).float()
    print(f"input {input}")

    # 创建隐藏层，4个输入，5个输出
    linear1 = nn.Linear(4, 5)

    # 加权求和
    l1 = linear1(input)
    print(f"l1 {l1}")

    # 激活函数
    output = torch.relu(l1)
    print(f"output {output}")

    # 神经元输出后随机失活，剩下的神经元加权
    dropout = nn.Dropout(p=0.4)
    output = dropout(output)
    print(f"output {output}")

if __name__ == '__main__':
    demo01()