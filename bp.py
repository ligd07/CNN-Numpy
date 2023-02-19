import numpy as np
from module import Layers


class BP(Layers):
    def __init__(self, name, in_channels, out_channels):
        """
            BP神经网络初始化

        Parameters
        ----------
        name:str
            BP神经网络层名称
        in_channels:int
            输出通道数
        out_channels:int
            输出通道数
        """
        super(BP, self).__init__(name)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weights = np.random.randn(self.out_channels, self.in_channels)
        self.bias = np.random.randn(self.out_channels, 1)

        self.grad_weights = np.random.randn(self.out_channels, self.in_channels)
        self.grad_bias = np.random.randn(self.out_channels, 1)

        self.data_input = np.zeros((self.in_channels, 1))

    def forward(self, data_in):
        """
            BP神经网络前向计算

        Parameters
        ----------
        data_in:numpy.ndarray
            输入数据，二维矩阵，大小为in_channels*1
        Returns
        -------
        data_out:numpy.ndarray
            输出数据，二维矩阵，大小为out_channels*1
        """
        self.data_input = data_in.reshape(self.in_channels, 1)
        data_out = self.weights @ self.data_input + self.bias
        return data_out

    def backward(self, grad_in):
        """
            BP神经网络误差反向传播计算

        Parameters
        ----------
        grad_in:numpy.ndarray
            输入误差梯度，二维矩阵，大小为out_channels*1
        Returns
        -------
        grad_out:numpy.ndarray
            输出误差梯度，二维矩阵，大小为in_channels*1
        """
        grad_out = self.weights.T @ grad_in

        self.grad_weights = grad_in @ self.data_input.T
        self.grad_bias = grad_in

        return grad_out

    def update(self, lr=1e-3):
        """
            参数更新

        Parameters
        ----------
        lr:float
            参数更新步长
        """
        self.weights -= lr * self.grad_weights
        self.bias -= lr * self.grad_bias
