import numpy as np

from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.layers import GRU, BatchNormalization

X_train = np.load('X_train.dat')  # 加载数据集和训练集
Y_train = np.load('Y_train.dat')
X_test = np.load('X_test.dat')
Y_test = np.load('Y_test.dat')

print("X_train.shape:", X_train.shape, "   ", "X_test.shape:", X_test.shape)
print("Y_train.shape:", Y_train.shape, "   ", "Y_test.shape:", Y_test.shape)

X_intput = Input(shape=(None, 39))

X = GRU(units=128)(X_intput)
X = BatchNormalization()(X)
X = Dropout(0.2)(X)

model_output = Dense(10, activation='softmax')(X)

model = Model(inputs=X_intput, outputs=model_output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

model.fit(X_train, Y_train, batch_size=32, epochs=20)

model.summary()

preds = model.evaluate(x=X_test, y=Y_test)
print("loss=" + str(preds[0]))
print("Test Accuracy=" + str(preds[1]))
