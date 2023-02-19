import numpy as np
from bp import BP
from module import Module
from activate import Sigmoid
import matplotlib.pyplot as plt
import datetime


# 定义误差函数
def loss(data_pred, data_true):
    data_out = ((data_true - data_pred) ** 2).mean()
    grad_out = 2 * (data_pred - data_true)
    return data_out, grad_out


def train(bp, x, y):
    e = 0
    for i in range(y.size):
        # 计算神经网络输出
        y_pred = bp.forward(x[i])

        # 计算损失与梯度
        y_loss, y_grad = loss(y_pred, y[i])

        # 反向传播
        bp.backward(y_grad)

        # 更新参数
        bp.step(0.1)

        e += y_loss

    return e


# 定义训练数据集
def train_data():
    data = np.random.random((300, 2))
    all_y_trues = np.zeros(300)
    for i in range(300):
        if np.sin(8 * data[i, 0]) > data[i, 1]:
            all_y_trues[i] = 0.0
        else:
            all_y_trues[i] = 1.0

    return data, all_y_trues


def test_plot(nn, x, y):
    x0 = list()
    x1 = list()
    # 依据y对x进行分类
    for i in range(300):
        if y[i] == 0:
            x0.append(x[i, :])
        else:
            x1.append(x[i, :])
    x0 = np.array(x0)
    x1 = np.array(x1)

    # 绘制分类后的数据
    plt.plot(x0[:, 0], x0[:, 1], 'ro')
    plt.plot(x1[:, 0], x1[:, 1], 'bo')

    # 生成测试数据坐标
    test_x, test_y = np.meshgrid(np.arange(-0.1, 1.1, 0.01), np.arange(-0.1, 1.1, 0.01))
    test_xy = np.array([test_x.ravel(), test_y.ravel()])
    test_z = np.zeros(test_xy.shape[1])

    # 测试
    for i in range(test_xy.shape[1]):
        test_z[i] = nn.forward(test_xy[:, i:i + 1])
    test_z = test_z.reshape(test_x.shape)

    # 绘制测试结果
    plt.contourf(test_x, test_y, test_z)


# 测试BP神经网络
if __name__ == '__main__':
    # 定义BP神经网络结构
    layers = [
        BP(name="BP1", in_channels=2, out_channels=3),
        Sigmoid(name="sigmoid1"),
        BP(name="BP2", in_channels=3, out_channels=1),
        Sigmoid(name="sigmoid2")
    ]
    # 实例化神经网络
    nn = Module(layers)

    # 定义训练数据集
    data, all_y_trues = train_data()

    # 记录训练误差
    e = []

    # 训练神经网络
    epochs = 1000  # number of times to loop through the entire dataset

    print('\r\n开始训练:', datetime.datetime.now())

    for epoch in range(epochs):

        train_e = train(nn, data, all_y_trues)
        e.append(train_e)

        if epoch % 10 == 0:
            r = round(epoch / epochs * 100, 2)
            print(f'\r训练已完成练{r}%', end='')

    print('\r训练已完成练100%')
    print('结束训练:', datetime.datetime.now())

    # 绘制测试结果
    test_plot(nn, data, all_y_trues)

    # 绘制误差曲线
    plt.figure(2)
    plt.plot(e)
    plt.show()
