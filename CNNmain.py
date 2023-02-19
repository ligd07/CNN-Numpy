import numpy as np
from conv import Conv2D
from module import Module
from activate import Sigmoid, SoftMax
from pool import MaxPool
from load_mnist import load_mnist
from bp import BP
import matplotlib.pyplot as plt
import datetime
from saveandread import save_cnn, read_cnn


# 损失函数与梯度
def loss(data_pred, data_true):
    data = np.zeros_like(data_pred)
    data[int(data_true), 0] = 1
    data_out = ((data - data_pred) ** 2).mean()
    grad_out = (data_pred - data)
    return data_out, grad_out


# 测试神经网络正确率
def test_check(cnn, images, labels):
    right = 0
    for i in range(labels.size):
        y = cnn.forward(images[i:i + 1, 0:1] / 255)
        if labels[i] == y.argmax():
            right += 1

        if (i + 1) % 10 == 0:
            print(f'\r测试已完成{round(i / 100, 2)}%', end='')

    print('\r测试已完成100%')
    return right / labels.size


def train(cnn, images, labels):
    for i in range(labels.size):
        # 计算神经网络输出
        y_pred = cnn.forward(images[i:i + 1, 0:1] / 255)

        # 计算损失与梯度
        y_loss, y_grad = loss(y_pred, labels[i])

        # 反向传播
        cnn.backward(y_grad)

        # 更新参数
        cnn.step(0.1)

        if (i + 1) % 20 == 0:
            print(f'\r训练已完成{round(i / 600, 4)}%', end='')

    print('\r训练已完成100%')


# 测试卷积神经网络功能
if __name__ == '__main__':
    # 定义神经网络结构，两层卷积层提取特征，一层展平层，两层BP神经网络分类
    layers = [
        Conv2D(name='conv1', in_shape=(1, 1, 28, 28), out_shape=(1, 6, 26, 26)),  # 6*26*26
        MaxPool(name='pool1'),  # 6*13*13
        Sigmoid(name="sigmoid1"),
        Conv2D(name='conv2', in_shape=(1, 6, 13, 13), out_shape=(1, 12, 10, 10)),  # 12*10*10
        MaxPool(name='pool2'),  # 12*5*5
        Sigmoid(name="sigmoid2"),
        Conv2D(name='conv3', in_shape=(1, 12, 5, 5), out_shape=(1, 100, 1, 1)),
        Sigmoid(name="sigmoid3"),
        BP(name="BP1", in_channels=100, out_channels=50),
        Sigmoid(name="sigmoid5"),
        BP(name="BP2", in_channels=50, out_channels=10),
        SoftMax(name="softmax1")
    ]

    # 实例化神经网络
    nn = Module(layers)

    # 读取训练数据集
    train_images, train_labels = load_mnist('./data/mnist')
    train_i_shape = train_images.shape
    train_images = train_images.reshape(train_i_shape[0], 1, train_i_shape[1], train_i_shape[2])

    # 读取测试数据集
    test_images, test_labels = load_mnist('./data/mnist', 't10k')
    test_i_shape = test_images.shape
    test_images = test_images.reshape(test_i_shape[0], 1, test_i_shape[1], test_i_shape[2])

    # 记录正确率曲线
    right_list = []

    # 训练并测试神经网络
    for epoch in range(10):
        # 训练
        print(f'\r\n第{epoch+1}次训练开始:', datetime.datetime.now())
        train(nn, train_images, train_labels)
        print(f'第{epoch + 1}次训练结束:', datetime.datetime.now())

        # 测试
        right_rate = test_check(nn, test_images, test_labels)
        right_rate = round(right_rate * 100, 2)
        print(f'第{epoch + 1}次测试结束:', datetime.datetime.now())
        print(f'第{epoch + 1}次训练正确率:{right_rate}% \r\n')

        right_list.append(right_rate)

        # 保存训练结果
        save_cnn(nn, filename=f'./parameters/第{epoch + 1}次训练参数-正确率{right_rate}%')

    # 绘制正确率曲线
    plt.plot(right_list)
    plt.show()
