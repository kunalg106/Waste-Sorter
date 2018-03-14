import numpy as np
import matplotlib.pyplot as plt
import math

# from scipy import misc
# from skimage.transform import resize
from random import shuffle  # Set seed
import tensorflow as tf
# import tensorflowvisu
from Dataset import Dataset

PATH = 'dataset'
num_train = 2100
img_size = (100,100,3)
img_len = img_size[0]*img_size[1]*img_size[2]
batch_size = 50
num_epochs = 20

data_set = Dataset(PATH, num_train, img_size)
print('Here')
X_test, Y_test = data_set.get_test_data()
# print(X_test.shape,Y_test.shape)
X = tf.placeholder(tf.float32, [None, img_size[0], img_size[1], img_size[2]])
Y_ = tf.placeholder(tf.float32, [None, 6])
lr = tf.placeholder(tf.float32)
# Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
pkeep = tf.placeholder(tf.float32)
# # five layers and their number of neurons (tha last layer has 10 softmax neurons)
# L = 200
# M = 100
# N = 60
# O = 30
# # Weights initialised with small random values between -0.2 and +0.2
# # When using RELUs, make sure biases are initialised with small *positive* values for example 0.1 = tf.ones([K])/10
# W1 = tf.Variable(tf.truncated_normal([img_len, L], stddev=0.1))  # 784 = 28 * 28
# B1 = tf.Variable(tf.ones([L])/10)
# W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
# B2 = tf.Variable(tf.ones([M])/10)
# W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
# B3 = tf.Variable(tf.ones([N])/10)
# W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
# B4 = tf.Variable(tf.ones([O])/10)
# W5 = tf.Variable(tf.truncated_normal([O, 6], stddev=0.1))
# B5 = tf.Variable(tf.ones([6])/10)

# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
K = 4  # first convolutional layer output depth
L = 8  # second convolutional layer output depth
M = 12  # third convolutional layer
N = 200  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([6, 6, 3, K], stddev=0.1))  # 5x5 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.ones([K])/10) # 97x97x4
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.ones([L])/10) #48x48x8
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.ones([M])/10) #24x24x12

W4 = tf.Variable(tf.truncated_normal([25 * 25 * M, N], stddev=0.1))
B4 = tf.Variable(tf.ones([N])/10)
W5 = tf.Variable(tf.truncated_normal([N, 6], stddev=0.1))
B5 = tf.Variable(tf.ones([6])/10)


# The model
stride = 1  # output is 28x28
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
stride = 2  # output is 14x14
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 2  # output is 7x7
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 25 * 25 * M])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
YY4 = tf.nn.dropout(Y4, pkeep)
Ylogits = tf.matmul(YY4, W5) + B5
Y = tf.nn.softmax(Ylogits)


# XX = tf.reshape(X, [-1, img_len])
#
# Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1)
# Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
# Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
# Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
# Ylogits = tf.matmul(Y4, W5) + B5
# Y = tf.nn.softmax(Ylogits)

# Ylogits = tf.matmul(XX, W) + b
#
# Y = tf.nn.softmax(Ylogits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = Ylogits, labels = Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# cross_entropy = -tf.reduce_mean(Y_*tf.log(Y))*1000.0

correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# batch_X, batch_Y = data_set.get_next_batch(50)
# print(batch_X.size, batch_Y.size)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

def train_model(num_epochs, batch_size):
    epoch = 1
    i = 0
    print("Starting epoch :", epoch)
    while (epoch <= num_epochs):
        if(data_set.is_end_of_list(batch_size)):
            epoch +=1
            i += 1
            data_set.reset_train_index()
            print("Training set exhausted. Starting epoch :", epoch)

        # learning rate decay
        max_learning_rate = 0.003
        min_learning_rate = 0.0001
        decay_speed = 20.0
        learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

        batch_X, batch_Y = data_set.get_next_batch(batch_size)
        acc_train, cross_train = sess.run([accuracy, cross_entropy], feed_dict = {X: batch_X, Y_: batch_Y, pkeep: 1.0})
        nun= sess.run(train_step, feed_dict = {X: batch_X, Y_: batch_Y, lr :learning_rate, pkeep: 0.75})
        acc_test, cross_test = sess.run([accuracy, cross_entropy], feed_dict = {X: X_test, Y_: Y_test, pkeep: 1.0})
        print("Train Accuracy: ", "%0.2f" % acc_train,"Train cross_entropy: ","%0.2f" %  cross_train, \
        "Test Accuracy: ", "%0.2f" %  acc_test, "Test cross_entropy: ", "%0.2f" % cross_test, "Learning rate: ", learning_rate)
        # tf.Print(X,[X])
train_model(num_epochs, batch_size)
