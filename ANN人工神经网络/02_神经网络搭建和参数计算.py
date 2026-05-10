import torch
import torch.nn as nn
from torchsummary import summary

'''
1. 第1个隐藏层：权重初始化采用标准化的xavier初始化 激活函数使用sigmoid
2. 第2个隐藏层：权重初始化采用标准化的He初始化 激活函数采用relu
3. out输出层线性层 假若多分类，采用softmax做数据归一化

'''
# todo: 搭建神经网络，自定义继承nn.Module
class NetDemo(nn.Module):
    def __init__(self):

        super().__init__()
        self.layer1 = nn.Linear(3, 3)
        self.layer2 = nn.Linear(3, 2)
        self.out = nn.Linear(2, 2)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.zeros_(self.layer1.bias)

        nn.init.kaiming_normal_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)

    def forward(self, x):

        x = self.layer1(x) # 加权求和
        x = torch.sigmoid(x) #激活函数

        x = self.layer2(x)
        x = torch.relu(x)

        x = torch.softmax(self.out(x), dim=-1)

        return x

def train():
    net = NetDemo()

    data = torch.randn(size=(5, 3))
    print(f"data:{data}, data.shape:{data.shape}, data.requires_grad:{data.requires_grad}")

    output = net(data)
    print(f"output:{output}, output.shape:{output.shape}, output.requires_grad:{output.requires_grad}")

    summary(net, (5, 3), device='cpu')

    for name, param in net.named_parameters():
        print(f"name:{name}, param:{param}")

if __name__ == '__main__':
    train()

