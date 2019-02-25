import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tf_utils import random_mini_batches


def create_placeholders(n_x, n_y):
    X = tf.placeholder(shape=[n_x, None], dtype='float')
    Y = tf.placeholder(shape=[n_y, None], dtype='float')

    return X, Y


def initialize_parameters():
    W1 = tf.get_variable("W1", [411, 3861], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [411, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [55, 411], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [55, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [10, 55], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [10, 1], initializer=tf.zeros_initializer())

    tf.add_to_collection(tf.GraphKeys.WEIGHTS, W1)  # 正则化
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, W2)
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, W3)

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X), b1)  # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)  # Z3 = np.dot(W3,Z2) + b3

    return Z3


def compute_cost(Z3, Y, reg_term):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    # 计算损失函数，reg_term是正则项
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)) + reg_term

    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.003,
          num_epochs=450, minibatch_size=32, print_cost=True):
    ops.reset_default_graph()
    (n_x, m) = X_train.shape  # n_x输入数据的维度，m训练样本的个数
    n_y = Y_train.shape[0]  # n_y输出数据的维度
    costs = []  # 保存损失函数的值

    # 创建placeholders
    X, Y = create_placeholders(n_x, n_y)

    # 初始化参数
    parameters = initialize_parameters()
    # 加入正则化
    regularizer = tf.contrib.layers.l2_regularizer(scale=912.816 / 33808)
    reg_term = tf.contrib.layers.apply_regularization(regularizer)

    # 神经网络前向传播
    Z3 = forward_propagation(X, parameters)

    # 就算损失函数
    cost = compute_cost(Z3, Y, reg_term)

    # 后向传播，优化参数，使用AdamOptimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    # 启动session
    with tf.Session() as sess:

        sess.run(init)

        # 训练迭代
        for epoch in range(num_epochs):

            epoch_cost = 0.  # 每次迭代的cost
            num_minibatches = int(m / minibatch_size)  # minibatches的数量
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch

                # 启动session执行optimizer，cost
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                epoch_cost += minibatch_cost / num_minibatches

            # 输出每50次迭代的cost
            if print_cost == True and epoch % 50 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # 画出反映cost趋势的图表
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # 保存最后优化参数
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # 计算正确的预测
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # 计算准确度
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters
