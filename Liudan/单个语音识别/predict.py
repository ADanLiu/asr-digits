import numpy as np
from mfcc_sec import get_wav_mfcc
from tf_utils import predict

audio = "E:\\corpus\\digitTrunk\\9-0683.wav"  # wav语音文件的路径
featureVex = get_wav_mfcc(audio).T  # 获取特征向量

parameters = np.load('parameters.npy').item()  # 获取训练完成的参数

prediction = predict(featureVex, parameters)
audioclass = ["eight", "five", "four", "nine", "one", "seven", "six", "three", "two", "zero"]  # 识别结果的种类的对照表
print("Audio recognition result is:", audioclass[prediction[0]])
