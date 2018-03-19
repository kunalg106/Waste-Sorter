import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from skimage.transform import resize
from random import shuffle
from shutil import copyfile

class Dataset(object):
    # This class contains all the methods which maybe required for
    # creating a dataset. Its attributes and several methods are private

    def __init__(self, PATHTrain, PATHTest, PATHConfu, imgSize):
        self.__PATHTrain = PATHTrain  # Path to the dir containing training datapoints
        self.__PATHTest =  PATHTest   # Path to the dir containing testing datapoints
        self.__PATHConfu = PATHConfu  # Path to the dir containing Confusion datapoints
        self.__imgSize = imgSize      # Image size of the datapoints
        self.__trainList = self.__getDatasetList(self.__PATHTrain)  # List of datapoints being used for training
        self.__testList = self.__getDatasetList(self.__PATHTest)  # List of datapoints being used for testing
        self.__trainIdx = 0  # Index upto which training datapoints have been extracted from trainList
        pass


    def __getDatasetList(self, PATH):
        # This method searches the entire dir provided in the path and creates
        # a complete list containing all the complete dataset
        labelNames = [label for label in os.listdir(PATH) if label[0] != '.']
        datasetList = list()
        for label in labelNames:
            classPath = os.path.join(PATH, label)
            # print("Loading Image from ", label)
            for img in os.listdir(classPath):
                if(img[0] != '.'):
                    datasetList.append((img,label))

        shuffle(datasetList)
        return datasetList

    def __readImgFromList(self, datasetList, PATH):
        # This method reads the image from the disk as per the provided list
        labelNames = [label for label in os.listdir(PATH) if label[0] != '.']
        # The following is the mapping being used to convert labels into one hot code
        oneHotCode =   {'cardboard': [1,0,0,0,0], \
                        'plastic'  : [0,1,0,0,0], \
                        'metal'    : [0,0,1,0,0], \
                        'paper'    : [0,0,0,1,0], \
                        'glass'    : [0,0,0,0,1]}

        X = np.zeros((len(datasetList), self.__imgSize[0], self.__imgSize[1], self.__imgSize[2]))
        Y = np.zeros((len(datasetList), 5))

        idx = 0
        for name, label in datasetList:
            classPath = os.path.join(PATH, label)
            imgPath = os.path.join(classPath, name)
            X[idx, :] = np.load(imgPath)
            Y[idx, :] = oneHotCode[label]
            idx += 1

        return X, Y

    def getNextBatch(self, batchSize):
        # This method reads the next batch of trainig images from the
        # train list and returns the corresponding data and label arrays
        X, Y = self.__readImgFromList(self.__trainList[self.__trainIdx:self.__trainIdx + batchSize], self.__PATHTrain)
        self.__trainIdx += batchSize

        return X, Y

    def hasNext(self, batchSize):
        # This method checks for the availability of another batch of training datapoints
        if(self.__trainIdx + batchSize < len(self.__trainList)):
            return True
        else:
            return False

    def resetTrainIdx(self):
        # Reset the train index to start a new epoch
        self.__trainIdx = 0

    def getTestData(self):
        # This method returns the test data and label arrays
        return self.__readImgFromList(self.__testList, self.__PATHTest)

    def getConfusionImg(self, label):
        # This method returns the datapoints of the given label for contructing confusion matrix
        classPath = os.path.join(self.__PATHConfu, label)
        datasetList = list()

        for img in os.listdir(classPath):
            if(img[0] != '.'):
                datasetList.append((img, label))

        return self.__readImgFromList(datasetList, self.__PATHConfu)

if __name__ == '__main__':

PATHConfu = 'datasetConfu'
PATHTest = 'datasetTest'
PATHTrain = 'datasetTrain'
imgSize = (100,100,3) # standard image size used for training
dataset = Dataset(PATHTrain, PATHTest, PATHConfu, imgSize)
Xtest, Ytest = dataset.getTestData()
XConfu, YConfu = dataset.getConfusionImg('glass')
XTrain, YTrain = dataset.getNextBatch(190)
# print(dataset.hasNext(2000))
# print(np.shape(XTrain), np.shape(YTrain))
# print(np.shape(XConfu), np.shape(YConfu))
# print(np.shape(Xtest), np.shape(Ytest))

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    # print('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush() # As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)
