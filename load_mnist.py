import numpy as np
import matplotlib.pyplot as plt
import struct


def load_mnist(path, kind='train'):
    """
    读取指定路径下的mnist数据集，并以ndarray数组得到形式分别返回数据集中的图像以及标签.

    Parameters
    ----------
    path:str
        数据集路径.
    kind:str
        读取数据集类型，默认读取训练数据集.
        kind='train':读取训练数据集.
        kind='t10k':读取测试数据集.

    Returns
    -------
    images,labels:tuple[np.ndarray, np.ndarray]
        图像,标签.

    Examples
    --------
    >>> train_img, train_lab = load_mnist('./data/mnist')
    >>> test_img, test_lab = load_mnist('./data/mnist', 't10k')
    """

    # 打开图片文件
    with open(path + "/" + kind + "-images.idx3-ubyte", 'rb') as images_path:
        # 读取文件
        images_data = np.fromfile(images_path, dtype=np.uint8)

    # 图片文件前16个字节为数据集信息,分别以int32类型,按照大端模式存储文件类型magic,数据集图片数量num,图片大小rows,cols.
    magic, num, rows, cols = struct.unpack('>IIII', images_data[0:16])
    # 将读取的图片信息按照解码的尺寸重新排列并将其数据类型变更为float.
    images = images_data[16:].reshape(num, rows, cols).astype("float32")

    # 打开标签文件
    with open(path + "/" + kind + "-labels.idx1-ubyte", 'rb') as labels_path:
        # 读取文件
        labels_data = np.fromfile(labels_path, dtype=np.uint8)
    # 图片文件前8个字节为数据集信息,分别以int32类型,按照大端模式存储文件类型magic,数据集标签数量num.
    magic, num = struct.unpack('>II', labels_data[0:8])
    # 将读取的标签信息按照解码的尺寸重新排列并将其数据类型变更为float.
    labels = labels_data[8:].reshape(num, ).astype("float32")

    return images, labels


if __name__ == '__main__':
    """
    读取训练集与测试集，并显示各数据集前4个图片及标签
    """

    # 读取训练数据集数据
    train_images, train_labels = load_mnist('./data/mnist')
    # 读取测试数据集数据
    test_images, test_labels = load_mnist('./data/mnist', 't10k')

    for i in range(4):
        # 在界面第1行显示4幅训练数据集图片及标签
        plt.subplot(2, 4, i + 1)
        plt.imshow(train_images[i])
        plt.axis('off')
        plt.title('train-' + str(train_labels[i]))

        # 在界面第2行显示4幅测试数据集图片及标签
        plt.subplot(2, 4, i + 5)
        plt.imshow(test_images[i])
        plt.axis('off')
        plt.title('test-' + str(test_labels[i]))

    plt.show()
