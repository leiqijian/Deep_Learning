'''

'''

# 导入相关模块
import torch
from torch.utils.data import TensorDataset   # 构造数据集对象
from torch.utils.data import DataLoader      # 数据加载器
from torch import nn                         # nn模块中有平方损失函数和假设函数
from torch import optim                      # optim模块中有优化器函数
from sklearn.datasets import make_regression # 创建线性回归模型数据集
import matplotlib.pyplot as plt


def create_dataset():
    # X : ndarray of shape
    # y : ndarray of shape
    # coef : ndarray of shape

    #借助make_regression，创建numpy数据集。
    #numpy --> Tensor (张量) --> TensorDataset (数据集对象) --> DataLoader (数据加载器)。目标是返回数据加载器
    x, y, coef = make_regression(
        n_samples=100,  #定义100条样本
        n_features=1,   #定义100个标签
        noise=10,       #定义噪声，噪声越大，样本点越散，噪声越小，样本点越集中
        coef=True,      #是否返回系数
        bias= 14.5,     #定义偏置
        random_state=3  #指定随机种子
    )

    #把 x , y 从numpy转成张量的形式返回
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)

    return x, y, coef

def train(x, y, coef):
    # 1. 创建数据集对象，把tensor --> 数据集对象 --> 数据加载器
    dataset = TensorDataset(x, y)

    # 2. 创建数据加载器对象
    # 参1：数据集对象，2.批次大小，3.是否打乱数据
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 3. 创建初始的线性回归模型
    # 在该案例中，只有一个特征和一个标签值
    model = nn.Linear(1, 1)

    # 4. 创建损失函数对象
    criterion = nn.MSELoss()

    # 5. 创建优化器对象
    # 1. 模型参数 2. 学习率
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 6. 具体的训练过程
    # 6.1 定义变量，分别表示：训练轮数，每轮（平均）损失值， 训练总损失，训练样本数
    epochs, loss_list, total_loss, total_sample = 5000, [], 0.0, 0
    # 6.2 开始训练
    for epoch in range(epochs): #epoch值：0， 1，2，3...99
        # 6.3 每轮分批次训练
        for train_x, train_y in dataloader: #100条数据，每批次16条，一共7批
            # 6.4 模型预测
            y_pred = model(train_x)

            loss = criterion(y_pred, train_y.reshape(-1, 1)) #把标签值转成列

            total_loss += loss.item()
            total_sample += 1
            # 梯度清零 + 反向传播 + 梯度更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_list.append(total_loss / total_sample)
        print(f"第{epoch + 1}轮，平均损失{total_loss / total_sample}")
    print(f"第{epochs}轮，平均损失分别为{total_loss / total_sample}")
    print(f"权重：{model.weight}， 偏置{model.bias}")





if __name__ == '__main__':
    x, y, coef = create_dataset()
    train(x, y, coef)
    print(x, y, coef)

