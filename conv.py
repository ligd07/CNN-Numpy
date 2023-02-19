import numpy as np
import scipy.signal as sig
from module import Layers


def conv4d(k, x, mode="valid"):
    """
        计算四维数组卷积，当卷积核有m*p个，输入数据有大小为p*n时，卷积后输出大小为m*n。
        若k=|k11 k12|,x=|x1|,则conv(k,x)=|k11*x1+k12*x2|
            |k21 k22|   |x2|             |k21*x1+k22*x2|

        输入的x的大小为2*1，表示输入数据有2个通道，输出大小为2*1.表示输出数据也有2个通道。

        但是对于大小为a*b*c*d四维数组A，a*b代表数据有a批，每批数据包含b个通道，每个通道是一个c*d大小的矩阵。

        所以将四维数组A进行conv4d运算时，需要将数组的0,1轴进行转置，使之符合第0维度代表通道数，第1维度代表批数的标准。

        进行运算后，结果依然是第0维度代表通道数，第1维度代表批数，若使输出变回第1维度代表通道数，第0维度代表批数，
        则输出后需要进行转置。

    Parameters
    ----------
    k:numpy.ndarray
        卷积核，四维数组，大小为m*p*l*l
    x:numpy.ndarray
        卷积对象，四维数组，大小为p*n*k*k
    mode:str
        卷积模式选择，valid模式或full模式

    Returns
    -------
    out:numpy.ndarray
        卷积输出，四维数组，大小为m*n*h*h，当mode='valid'时，h=k-l+1。当mode='full'时，h=k+l-1
    """
    x_shape = x.shape
    k_shape = k.shape
    if mode == 'valid':
        o_size = x_shape[2] - k_shape[2] + 1
    else:
        o_size = x_shape[2] + k_shape[2] - 1
    out = np.zeros((k_shape[0], x_shape[1], o_size, o_size))

    for i in range(k_shape[0]):
        for j in range(x_shape[1]):
            for h in range(k_shape[1]):
                out[i, j] += sig.convolve2d(k[i, h], x[h, j], mode=mode)

    return out


class Conv2D(Layers):
    def __init__(self, name, in_shape, out_shape, mode='valid'):
        """
            卷积层初始化

        Parameters
        ----------
        name:str
            卷积层名称
        out_shape:tuple
            卷积层输出数据的形状(1*m*h*h)
        in_shape:tuple
            卷积层输入数据的形状(1*n*l*l)
        mode:str
            卷积模式，valid或full
        """
        super(Conv2D, self).__init__(name)

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.mode = mode

        if self.mode == 'valid':
            self.kernel_size = self.in_shape[2] - self.out_shape[2] + 1
        else:
            self.kernel_size = self.out_shape[2] - self.in_shape[2] + 1

        self.weights = np.random.randn(self.out_shape[1], self.in_shape[1], self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_shape)

        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
        self.data_input = np.zeros(self.in_shape)

    def forward(self, data_in):
        """
            计算卷积层前向输出

        Parameters
        ----------
        data_in:numpy.ndarray
            输入数据，四维数组，大小为(1*n*l*l)
        Returns
        -------
        data_out:numpy.ndarray
            输出数据，四维数组，大小为(1*m*h*h)
        """
        self.data_input = data_in.transpose((1, 0, 2, 3))
        data_out = conv4d(self.weights, self.data_input, mode=self.mode).transpose((1, 0, 2, 3)) + self.bias
        return data_out

    def backward(self, grad_in):
        """
            误差传播反向计算，并计算卷积核与偏移的梯度

        Parameters
        ----------
        grad_in:numpy.ndarray
            输入的误差梯度
        Returns
        -------
        grad_out:numpy.ndarray
            输出的误差梯度
        """
        k_rtt = np.rot90(self.weights, 2, (2, 3)).transpose((1, 0, 2, 3))
        in_rtt = np.rot90(self.data_input, 2, (2, 3)).transpose((1, 0, 2, 3))
        grad_in_t = grad_in.transpose((1, 0, 2, 3))
        if self.mode == 'valid':
            grad_out = conv4d(k_rtt, grad_in_t, mode='full').transpose((1, 0, 2, 3))
        else:
            grad_out = conv4d(k_rtt, grad_in_t, mode='valid').transpose((1, 0, 2, 3))

        self.grad_weights = conv4d(grad_in_t, in_rtt, mode='valid')
        self.grad_bias = grad_in

        return grad_out

    def update(self, lr=1e-3):
        """
            更新卷积层参数

        Parameters
        ----------
        lr:float
            参数更新步长
        """
        self.weights -= lr * self.grad_weights
        self.bias -= lr * self.grad_bias


if __name__ == '__main__':
    np.random.seed(1)
    x = np.random.randint(10, size=(3, 2, 5, 5))
    conv = Conv2D('conv1', (3, 3, 2, 2), x.shape, mode='valid')
    y = conv.forward(x)
    print(y.shape)

    loss = y - (y + 1)
    grad = conv.backward(loss)

    print(grad.shape)
