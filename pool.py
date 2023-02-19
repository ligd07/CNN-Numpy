import numpy as np
from module import Layers


def max_pool(x, kernel_size=2, stride=2):
    """
    对一个二维矩阵进行最大值池化操作，返回池化后的矩阵与池化后元素对应输入矩阵的索引

    Parameters
    ----------
    x:numpy.ndarray
        池化对象，二维矩阵
    kernel_size:int
        池化核大小
    stride:int
        池化步长

    Returns
    -------
    out,index:tuple[np.ndarray, np.ndarray]
        池化后的输出，池化后输出值对应输入矩阵的位置

    Examples
    --------
    >>> a=np.array([[2,5,4,1],[3,6,8,4],[5,1,6,9],[5,2,7,9]])
    >>> b, c = max_pool(a)
    >>> a
    array([[2, 5, 4, 1],
           [3, 6, 8, 4],
           [5, 1, 6, 9],
           [5, 2, 7, 9]])
    >>> b
    array([[6., 8.],
           [5., 9.]])
    >>> c
    array([[0, 0, 0, 0],
           [0, 1, 1, 0],
           [1, 0, 0, 1],
           [0, 0, 0, 0]])
    """
    rows, cols = x.shape
    index = np.zeros_like(x)
    m = rows // stride
    n = cols // stride
    out = np.zeros((m, n))
    for i in range(m):
        row_start = i * stride
        row_end = row_start + kernel_size
        for j in range(n):
            col_start = j * stride
            col_end = col_start + kernel_size
            out[i, j] = x[row_start:row_end, col_start:col_end].max()
            max_index = x[row_start:row_end, col_start:col_end].argmax()
            index[row_start + max_index // kernel_size, col_start + max_index % kernel_size] = 1
    return out, index


class MaxPool(Layers):
    def __init__(self, name, kernel_size=2, stride=2):
        """
        池化层初始化

        Parameters
        ----------
        name:str
            池化层名称
        kernel_size:int
            池化核大小
        stride:int
            池化步长
        """
        super(MaxPool, self).__init__(name)
        self.kernel_size = kernel_size
        self.stride = stride
        self.index = np.array([0])

    def forward(self, x):
        """
            池化层前向输出计算

        Parameters
        ----------
        x:numpy.ndarray
            池化对象，四维数组
        Returns
        -------
        out:numpy.ndarray
            池化后的四维数组
        """
        n, c, rows, cols = x.shape
        self.index = np.zeros_like(x)
        o_rows = rows // self.stride
        o_cols = cols // self.stride
        out = np.zeros((n, c, o_rows, o_cols))

        for i in range(n):
            for j in range(c):
                out[i, j], self.index[i, j] = max_pool(x[i, j], kernel_size=self.kernel_size, stride=self.stride)

        return out

    def backward(self, grad_in):
        """
            误差反向传播时，依据输入的误差梯度计算输出的误差梯度

        Parameters
        ----------
        grad_in:numpy.ndarray
            误差反向传播时来自前一层的误差梯度
        Returns
        -------
        gard_out:numpy.ndarray
            误差反向传播时传递给后一层的误差梯度
        """
        gard_out = np.repeat(np.repeat(grad_in, self.stride, axis=2), self.stride, axis=3) * self.index
        return gard_out


if __name__ == '__main__':
    a = np.random.randint(20, size=(2, 2, 4, 4))
    pool = MaxPool('pool1')
    b = pool.forward(a)
    print(b)
