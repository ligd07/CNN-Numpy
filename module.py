class Layers:
    """
    神经网络层模板
    神经网络需要实现以下基本功能函数:
        forward:神经网络前向计算
        backward:神经网络误差反向传播，计算误差梯度
        update:更新神经网络参数
    """
    def __init__(self, name):
        self.name = name

    def forward(self, data_in):
        pass

    def backward(self, grad_in):
        pass

    def update(self, lr=1e-3):
        pass


class Module:
    """
    神经网络模板
    神经网络需要实现以下基本功能函数:
        forward:神经网络前向计算
        backward:神经网络误差反向传播，计算每层误差梯度
        step:更新神经网络所有参数
    """
    def __init__(self, layers):
        """
        依据给定的神经网络结构列表创建神经网络

        Parameters
        ----------
        layers:list
            神经网络结构列表
        """
        self.layers = layers

    def forward(self, x):
        """
        依据神经网络的结构与输入计算输出

        Parameters
        ----------
        x:numpy.ndarray
            神经网络的输入

        Returns
        -------
        x:numpy.ndarray
            神经网络的输出
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        """
        误差反向传播计算

        Parameters
        ----------
        grad:numpy.ndarray
            输出层误差梯度
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def step(self, lr=1e-3):
        """
        神经网络参数更新

        Parameters
        ----------
        lr:float
            参数更新步长
        """
        for layer in reversed(self.layers):
            layer.update(lr)
