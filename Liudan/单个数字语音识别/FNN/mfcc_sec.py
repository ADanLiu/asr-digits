import scipy.io.wavfile as wav
import numpy as np
from python_speech_features import mfcc
from python_speech_features import delta


def get_wav_mfcc(wavpath):
    fs, audio = wav.read(wavpath)
    processed_audio = mfcc(audio, fs)  # 求13维倒谱系数
    delta1 = delta(processed_audio, 1)  # 求13维Delta倒谱系数
    delta2 = delta(processed_audio, 2)  # 求13 维双Delta倒谱系数

    features = np.hstack((processed_audio, delta1, delta2))  # 合并以上特征，构成39维特征向量
    featureRow = features.reshape(1, -1)  # 将矩阵合并成行向量

    # 让所有的语音特征向量保持同样的维度
    featureVec = list(np.array(featureRow[0]))
    # print(len(featureVec))
    while len(featureVec) > 3861:
        del featureVec[len(featureRow[0]) - 1]
        del featureVec[0]
    # print(len(featureVec))
    while len(featureVec) < 3861:
        featureVec.append(0)
    featureVec = np.array(featureVec)
    featureVec = featureVec.reshape(1, -1)
    return featureVec
