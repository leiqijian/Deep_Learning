from torch import nn


def demo01():
    # 1. 创建1个线性层，输入维度5，输出维度3
    linear = nn.Linear(5, 3)
    # 2. 对权重w进行随机初始化， 从0-1均匀分布产生参数
    nn.init.uniform_(linear.weight)
    # 2. 对b进行随机初始化， 从0-1均匀分布产生参数
    nn.init.uniform_(linear.bias)
    # 相当于
    print(linear.weight)
    # tensor([[0.4735, 0.9235, 0.9520, 0.0903, 0.8395],
    #         [0.9274, 0.1535, 0.8977, 0.5273, 0.9717],
    #         [0.8190, 0.4586, 0.2328, 0.3724, 0.7459]], requires_grad=True)
    print(linear.weight.data)
    # tensor([[0.4735, 0.9235, 0.9520, 0.0903, 0.8395],
    #         [0.9274, 0.1535, 0.8977, 0.5273, 0.9717],
    #         [0.8190, 0.4586, 0.2328, 0.3724, 0.7459]])
    print(linear.bias)
    # tensor([0.3331, 0.2880, 0.6865], requires_grad=True)

if __name__ == '__main__':
    demo01()