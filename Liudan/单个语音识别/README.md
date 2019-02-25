文件说明：
mfcc_sec.py：获取语音数据的MFCC-39维特征
createDataset.py：创建训练集和测试集数据并保存
buildmodel.py：构建神经网络模型并训练
predict.py：使用训练好的模型进行识别其他单个数字语音
tf_utils.py：包含一些供调用的工具函数：one_hot_matrix，random_mini_batches，predict，forward_propagation_for_predict

模型参数：
三层神经元数量分别是：411,55,10
学习速率：0.003
正则化系数：912.816/33808
迭代次数：450
minibatch_size=32
训练语音数量：33808
测试语音数量：5000

训练效果：
训练准确度：0.99816614
测试准确度：0.903