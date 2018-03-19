import numpy as np
import matplotlib.pyplot as plt
import math
from random import shuffle  # Set seed
import tensorflow as tf
from Dataset import Dataset

PATHConfu = 'datasetConfu'
PATHTest = 'datasetTest'
PATHTrain = 'datasetTrain'
imgSize = (100,100,3) # standard image size used for training

oneHotCode =   {'cardboard': [1,0,0,0,0], \
                'plastic'  : [0,1,0,0,0], \
                'metal'    : [0,0,1,0,0], \
                'paper'    : [0,0,0,1,0], \
                'glass'    : [0,0,0,0,1]}

batchSize = 40
numEpochs = 250

dataset = Dataset(PATHTrain, PATHTest, PATHConfu, imgSize)
XTest, YTest = dataset.getTestData()

X = tf.placeholder(tf.float32, [None, imgSize[0], imgSize[1], imgSize[2]], name = "X")
Y_ = tf.placeholder(tf.float32, [None, 5], name = "Y_")
lr = tf.placeholder(tf.float32)
pkeep = tf.placeholder(tf.float32)



# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
K = 4  # first convolutional layer output depth
L = 8  # second convolutional layer output depth
M = 12  # third convolutional layer
N = 200  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([6, 6, 3, K], stddev=0.1), name = "W1")  # 5x5 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.ones([K])/10, name = "B1") # 97x97x4
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1), name = "W2")
B2 = tf.Variable(tf.ones([L])/10, name = "B2") #48x48x8
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1), name = "W3")
B3 = tf.Variable(tf.ones([M])/10, name = "B3") #24x24x12

W4 = tf.Variable(tf.truncated_normal([25 * 25 * M, N], stddev=0.1), name = "W4")
B4 = tf.Variable(tf.ones([N])/10, name = "B4")
W5 = tf.Variable(tf.truncated_normal([N, 5], stddev=0.1), name = "W5")
B5 = tf.Variable(tf.ones([5])/10, name = "B5")

# The model
with tf.name_scope("layer1"):
    stride = 1  # output is 28x28
    X = tf.nn.dropout(X, pkeep)
    Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
with tf.name_scope("layer2"):
    stride = 2  # output is 14x14
    Y1 = tf.nn.dropout(Y1, pkeep)
    Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
with tf.name_scope("layer3"):
    stride = 2  # output is 7x7
    Y2 = tf.nn.dropout(Y2,pkeep)
    Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 25 * 25 * M])

with tf.name_scope("layer4"):
    YY = tf.nn.dropout(YY, pkeep)
    Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
with tf.name_scope("layer5"):
    YY4 = tf.nn.dropout(Y4, pkeep)
    Ylogits = tf.matmul(YY4, W5) + B5
    Y = tf.nn.softmax(Ylogits)

tf.summary.histogram("W1", W1)
tf.summary.histogram("W2", W2)
tf.summary.histogram("W3", W3)
tf.summary.histogram("W4", W4)
tf.summary.histogram("W5", W5)

tf.summary.histogram("B1", B1)
tf.summary.histogram("B2", B2)
tf.summary.histogram("B3", B3)
tf.summary.histogram("B4", B4)
tf.summary.histogram("B5", B5)

with tf.name_scope("cost"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = Ylogits, labels = Y_)
    cross_entropy = tf.reduce_mean(cross_entropy)*100
    tf.summary.scalar("cost", cross_entropy)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

predictionCard2Card = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(oneHotCode['cardboard']), tf.argmax(Y,1)), tf.float32))
predictionCard2Plastic = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(oneHotCode['plastic']), tf.argmax(Y,1)), tf.float32))
predictionCard2Metal = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(oneHotCode['metal']), tf.argmax(Y,1)), tf.float32))
predictionCard2Paper = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(oneHotCode['paper']), tf.argmax(Y,1)), tf.float32))
predictionCard2Glass = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(oneHotCode['glass']), tf.argmax(Y,1)), tf.float32))

predictionPlastic2Card = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(oneHotCode['cardboard']), tf.argmax(Y,1)), tf.float32))
predictionPlastic2Plastic = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(oneHotCode['plastic']), tf.argmax(Y,1)), tf.float32))
predictionPlastic2Metal = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(oneHotCode['metal']), tf.argmax(Y,1)), tf.float32))
predictionPlastic2Paper = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(oneHotCode['paper']), tf.argmax(Y,1)), tf.float32))
predictionPlastic2Glass = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(oneHotCode['glass']), tf.argmax(Y,1)), tf.float32))

predictionMetal2Card = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(oneHotCode['cardboard']), tf.argmax(Y,1)), tf.float32))
predictionMetal2Plastic = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(oneHotCode['plastic']), tf.argmax(Y,1)), tf.float32))
predictionMetal2Metal = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(oneHotCode['metal']), tf.argmax(Y,1)), tf.float32))
predictionMetal2Paper = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(oneHotCode['paper']), tf.argmax(Y,1)), tf.float32))
predictionMetal2Glass = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(oneHotCode['glass']), tf.argmax(Y,1)), tf.float32))

predictionPaper2Card = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(oneHotCode['cardboard']), tf.argmax(Y,1)), tf.float32))
predictionPaper2Plastic = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(oneHotCode['plastic']), tf.argmax(Y,1)), tf.float32))
predictionPaper2Metal = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(oneHotCode['metal']), tf.argmax(Y,1)), tf.float32))
predictionPaper2Paper = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(oneHotCode['paper']), tf.argmax(Y,1)), tf.float32))
predictionPaper2Glass = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(oneHotCode['glass']), tf.argmax(Y,1)), tf.float32))


predictionGlass2Card = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(oneHotCode['cardboard']), tf.argmax(Y,1)), tf.float32))
predictionGlass2Plastic = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(oneHotCode['plastic']), tf.argmax(Y,1)), tf.float32))
predictionGlass2Metal = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(oneHotCode['metal']), tf.argmax(Y,1)), tf.float32))
predictionGlass2Paper = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(oneHotCode['paper']), tf.argmax(Y,1)), tf.float32))
predictionGlass2Glass = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(oneHotCode['glass']), tf.argmax(Y,1)), tf.float32))

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    writer = tf.summary.FileWriter("./logs/nn_logs", sess.graph)

    merged = tf.summary.merge_all()
    sess.run(init)

    def train_model(numEpochs, batchSize):
        epoch = 1
        i = 0
        idx = 0
        # Confu_set = data_set.get_confusion_img()

        print("Starting epoch :", epoch)
        while (epoch <= numEpochs):
            # if(data_set.is_end_of_list(batch_size)):
            if(dataset.hasNext(batchSize) == False):
                epoch +=1
                i += 1
                # data_set.reset_train_index()
                dataset.resetTrainIdx()
                print("Training set exhausted. Starting epoch :", epoch)

            # learning rate decay
            max_learning_rate = 0.0003
            min_learning_rate = 0.0001
            decay_speed = 200.0
            learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

            # batch_X, batch_Y = data_set.get_next_batch(batch_size)
            XTrain, YTrain = dataset.getNextBatch(batchSize)

            accTrain, crossTrain = sess.run([accuracy, cross_entropy], feed_dict = {X: XTrain, Y_: YTrain, pkeep: 1.0})

            nun = sess.run(train_step, feed_dict = {X: XTrain, Y_: YTrain, lr :learning_rate, pkeep: 0.6})

            if(idx %10 == 0):
                summaryTest, accTest, crossTest = sess.run([merged, accuracy, cross_entropy], feed_dict = {X: XTest, Y_: YTest, pkeep: 1.0})
                writer.add_summary(summaryTest, idx)
                print("Train Accuracy: ", "%0.2f" % accTrain,"Train cross_entropy: ","%0.2f" %  crossTrain, \
                "Test Accuracy: ", "%0.2f" %  accTest, "Test cross_entropy: ", "%0.2f" % crossTest, "Learning rate: ", "%0.5f" % learning_rate)

                XConfu, YConfu = dataset.getConfusionImg('cardboard')
                predCard2Card, predCard2Plastic, predCard2Metal, predCard2Paper, predCard2Glass = \
                sess.run([predictionCard2Card, predictionCard2Plastic, predictionCard2Metal, predictionCard2Paper, predictionCard2Glass], \
                feed_dict = {X: XConfu, pkeep: 1.0})

                XConfu, YConfu = dataset.getConfusionImg('plastic')
                predPlastic2Card, predPlastic2Plastic, predPlastic2Metal, predPlastic2Paper, predPlastic2Glass = \
                sess.run([predictionPlastic2Card, predictionPlastic2Plastic, predictionPlastic2Metal, predictionPlastic2Paper, predictionPlastic2Glass], \
                feed_dict = {X: XConfu, pkeep: 1.0})

                XConfu, YConfu = dataset.getConfusionImg('metal')
                predMetal2Card, predMetal2Plastic, predMetal2Metal, predMetal2Paper, predMetal2Glass = \
                sess.run([predictionMetal2Card, predictionMetal2Plastic, predictionMetal2Metal, predictionMetal2Paper, predictionMetal2Glass], \
                feed_dict = {X: XConfu, pkeep: 1.0})

                XConfu, YConfu = dataset.getConfusionImg('paper')
                predPaper2Card, predPaper2Plastic, predPaper2Metal, predPaper2Paper, predPaper2Glass = \
                sess.run([predictionPaper2Card, predictionPaper2Plastic, predictionPaper2Metal, predictionPaper2Paper, predictionPaper2Glass], \
                feed_dict = {X: XConfu, pkeep: 1.0})

                XConfu, YConfu = dataset.getConfusionImg('glass')
                predGlass2Card, predGlass2Plastic, predGlass2Metal, predGlass2Paper, predGlass2Glass = \
                sess.run([predictionGlass2Card, predictionGlass2Plastic, predictionGlass2Metal, predictionGlass2Paper, predictionGlass2Glass], \
                feed_dict = {X: XConfu, pkeep: 1.0})

                print("Confusion Matrix..........................................")
                print('cardboard', ["%0.2f" % predCard2Card,  "%0.2f" %   predCard2Plastic,  "%0.2f" %   predCard2Metal,  "%0.2f" %   predCard2Paper,  "%0.2f" %   predCard2Glass])
                print('plastic  ', ["%0.2f" % predPlastic2Card,"%0.2f" %  predPlastic2Plastic,"%0.2f" %  predPlastic2Metal, "%0.2f" % predPlastic2Paper,"%0.2f" %  predPlastic2Glass])
                print('metal    ', ["%0.2f" % predMetal2Card,  "%0.2f" %  predMetal2Plastic,  "%0.2f" %  predMetal2Metal, "%0.2f" %   predMetal2Paper, "%0.2f" %   predMetal2Glass])
                print('paper    ', ["%0.2f" % predPaper2Card,  "%0.2f" %  predPaper2Plastic,  "%0.2f" %  predPaper2Metal, "%0.2f" %   predPaper2Paper, "%0.2f" %   predPaper2Glass])
                print('glass    ', ["%0.2f" % predGlass2Card,  "%0.2f" %  predGlass2Plastic,  "%0.2f" %  predGlass2Metal, "%0.2f" %   predGlass2Paper, "%0.2f" %   predGlass2Glass])

            idx +=1

    train_model(numEpochs, batchSize)
