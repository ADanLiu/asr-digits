import os
import numpy as np
import random
from mfcc_sec import get_wav_mfcc
from tf_utils import one_hot_matrix

labsIndName = []
num_class = 0


def creatDataset():
    wavs = []  # 训练wav数据集
    labels = []  # lables和testlabel存放对应标签的下标，下标对应的名字存放在labsInd
    testwavs = []  # 测试wav数据集
    testlabels = []  # 测试集标签

    path = "D:\\corpus\\digits"
    dirs = os.listdir(path)  # 获取目录列表
    for i in dirs:
        print("开始加载：", i)
        labsIndName.append(i)  # 当前分类进入到标签的名字集
        wavpath = path + "\\" + i
        # testNum = 0  # 当前分类进入测试集的有几个，每个分类有100个进入测试集
        files = os.listdir(wavpath)  # 某个目录下所含文件的列表

        test_random_list = random.sample(files, 250)

        for j in files:
            try:
                wavFeature = get_wav_mfcc(wavpath + "\\" + j)  # 获取mfcc特征向量
                if j in test_random_list:  # 获取测试集
                    testwavs.append(wavFeature)  # 将每个测试语音的特征加入到测试数据集
                    testlabels.append(labsIndName.index(i))  # 获取当前语音的标签在labsIndName中的索引
                    # testNum += 1
                else:  # 获取训练集
                    wavs.append(wavFeature)  # 将每个训练语音的特征加入到训练数据集
                    labels.append(labsIndName.index(i))  # 获取当前语音的标签在labsIndName中的索引
            except:
                pass
    num_class = len(labsIndName)
    wavs = np.array(wavs)
    labels = np.array(labels)
    testwavs = np.array(testwavs)
    testlabels = np.array(testlabels)
    return (wavs, labels), (testwavs, testlabels), num_class


if __name__ == '__main__':
    (X_train, labels), (X_test, testlabels), num_class = creatDataset()
    # X_train = wavs.reshape(wavs.shape[0], -1).T  # 将维度为(a,b,c,d) 转换为(b ∗c ∗d, a)
    # X_test = testwavs.reshape(testwavs.shape[0], -1).T
    print(X_train.shape, "   ", X_test.shape)

    Y_train = one_hot_matrix(labels, num_class).T  # 将数据的label转换成one hot码
    Y_test = one_hot_matrix(testlabels, num_class).T
    print(Y_train.shape, "   ", Y_test.shape)

    X_train.dump('X_train.dat')  # 保存数据集到dat文件
    X_test.dump('X_test.dat')
    Y_train.dump('Y_train.dat')
    Y_test.dump('Y_test.dat')
