import numpy as np
from buildmodle import model

if __name__ == '__main__':
    X_train = np.load('X_train.dat')  # 加载数据集和训练集
    Y_train = np.load('Y_train.dat')
    X_test = np.load('X_test.dat')
    Y_test = np.load('Y_test.dat')
    print("X_train.shape:", X_train.shape, "   ", "X_test.shape:", X_test.shape)
    print("Y_train.shape:", Y_train.shape, "   ", "Y_test.shape:", Y_test.shape)

    parameters = model(X_train, Y_train, X_test, Y_test)  # 训练模型并获取最后的参数

    np.save('parameters.npy', parameters)  # 将训练好的参数保存到文件中，供predict调用
