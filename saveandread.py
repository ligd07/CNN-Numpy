import numpy as np
from conv import Conv2D
from bp import BP
from module import Module
from activate import Sigmoid, SoftMax
from pool import MaxPool
from load_mnist import load_mnist
import matplotlib.pyplot as plt


def save_cnn(cnn, filename='CNN_data'):
    """
        保存神经网络各层参数

    Parameters
    ----------
    cnn:Module
        需要保存参数的神经网络
    filename:str
        参数文件的路径及名称
    """
    d = {}

    for nc in cnn.layers:
        if isinstance(nc, (BP, Conv2D)):
            d[nc.name + '-weight'] = nc.weights
            d[nc.name + '-bias'] = nc.bias

    np.savez(filename, **d)


def read_cnn(cnn, filename='CNN_data.npz'):
    """
        从参数文件中读取数据并更新神经网络各层参数，需要保证参数文件与给定神经网络结构一致

    Parameters
    ----------
    cnn:Module
        需要从参数文件更新参数的神经网络
    filename:str
        参数文件的路径及名称
    """
    d = np.load(filename)
    for nc in cnn.layers:
        if isinstance(nc, (BP, Conv2D)):
            nc.weights = d[nc.name + '-weight']
            nc.bias = d[nc.name + '-bias']


if __name__ == '__main__':
    # 定义神经网络结构
    # 读取的是CNNmain.py中神经网络的参数数据，所以结构需与之一致
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

    # 读取参数文件
    read_cnn(nn, filename='./parameters/第10次训练参数-正确率96.98%.npz')

    # 读取测试文件集
    test_images, test_labels = load_mnist('./data/mnist', 't10k')
    test_i_shape = test_images.shape
    test_images = test_images.reshape(test_i_shape[0], 1, test_i_shape[1], test_i_shape[2])

    for i in range(4):
        # 在界面第1行显示4幅训练数据集图片及标签
        t = nn.forward(test_images[i:i + 1, 0:1] / 255).argmax()
        plt.subplot(2, 4, i + 1)
        plt.imshow(test_images[i, 0])
        plt.axis('off')
        plt.title('labels-' + str(test_labels[i])+'\nCNN-'+str(t))

        # 在界面第2行显示4幅测试数据集图片及标签
        t = nn.forward(test_images[i+4:i + 5, 0:1] / 255).argmax()
        plt.subplot(2, 4, i + 5)
        plt.imshow(test_images[i+4, 0])
        plt.axis('off')
        plt.title('labels-' + str(test_labels[i+4])+'\nCNN-'+str(t))

    plt.show()
