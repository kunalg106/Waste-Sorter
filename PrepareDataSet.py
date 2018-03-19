import os
import numpy as np
from scipy import misc
from skimage.transform import resize
from random import shuffle
from shutil import move


PATH0 = 'dataset-resized'  # Path to the original dataset directory
PATH = 'dataset-normalized' # Path to the normalized dataset
PATHConfu = 'datasetConfu'
PATHTest = 'datasetTest'
PATHTrain = 'datasetTrain'
imgSize = (100,100,3) # standard image size used for training
numTrain = 400
numTest = 120
numConfu = 20
def relabelDataSet():
    # This function is used to relabel the entire dataset to make it easier to read
    # For example an arbitrary datapoit file in the class cardboard say sfnjfnsn.jpg
    # will be converted into something like cardboard123.jpg

    # Gather a list of class labels
    labelNames = [label for label in os.listdir(PATH0) if label[0] != '.']

    for label in labelNames:
            classPath0 = os.path.join(PATH0, label) # Path to a single class dir
            idx = 1    # index of the datapoint
            for img in os.listdir(classPath0):  #listdir provides a list of all files in the directory
                if(img[0] != '.'):     # to avoid listing of hidden files
                    # Renamed file eg. cardboard123
                    os.rename(os.path.join(classPath0, img), os.path.join(classPath0, label) + str(idx))
                    idx+=1

def normalizeDataSet():
    # This function is used to normalize the datapoints by subtracting the mean image
    # of each class from their respective datapoints. Also each class is subjected
    # to normalization of their pixel values to boost training performance
    # This function creates a separate directory for the normalized dataset where
    # the datapoints are stored in numpy array format as opposed to the .jpg format

    # Gather a list of class labels
    labelNames = [label for label in os.listdir(PATH0) if label[0] != '.']

    XMean = np.zeros((imgSize[0], imgSize[1], imgSize[2])) # Mean image of a class
    X = np.zeros((imgSize[0], imgSize[1], imgSize[2])) # Current image being processed

    for label in labelNames:
        count = 0
        classPath0 = os.path.join(PATH0, label) # Path to a single class dir

        # Calculate the mean image for the given class label
        for img in os.listdir(classPath0): #listdir provides a list of all files in the directory
            if(img[0] != '.'):          # to avoid listing of hidden files
                print("Loading ", img)
                XMean += resize(misc.imread(os.path.join(classPath0, img)), imgSize)
                count +=1
        XMean = XMean/count

        # Create dirs for storing the normalized dataset.
        if not os.path.exists(os.path.join(PATH, label)):
            os.makedirs(os.path.join(PATH, label))
        classPath = os.path.join(PATH, label)  # Path the a single class in normalized dataset

        # Store the normalized datapoints of the given label
        for img in os.listdir(classPath0):
            if(img[0] != '.'):
                print("Saving ", img)
                X = resize(misc.imread(os.path.join(classPath0, img)), imgSize)
                X -= XMean    # Subtract given Image from the mean image
                # Store this Image as a numpy array file in the normalized dataset dir
                np.save(os.path.join(classPath, img), X)

def partitionDataSet(numTrain, numTest, numConfu):
    # This function partisions the dataset into three separate directories which
     # contain respective training, testing and confusion datapoints.


     # Gather a list of class labels
     labelNames = [label for label in os.listdir(PATH) if label[0] != '.']

     for label in labelNames:
         classPath = os.path.join(PATH, label) # Path the a single class in normalized dataset

         datasetList = list() # list of datapoints in a given class is stored here
         trainList = list()   # List of training datapoints in a given class
         testList = list()    # List of testing datapoints in a given class
         confuList = list()   # List of confusion datapoints in a given class

         # Append all the files in sub-dir to the main list along with their class name
         for img in os.listdir(classPath):
             if(img[0] != '.'):
                datasetList.append(img)

         shuffle(datasetList) # Shuffle the entire list to remove any bias
         confuList = datasetList[:numConfu]  # make list of confusion datapoints
         testList = datasetList[numConfu:numConfu+numTest] # make list of test datapoints
         trainList = datasetList[numConfu+numTest:numConfu+numTest+numTrain] # make list of train datapoints

         # Creating separate dir for confusion datapoints
         for img in confuList:
             # Path to the dir containing confusion datapoints of the given class label
            classPathConfu = os.path.join(PATHConfu, label)

            # Create dirs for storing the confusion dataset.
            if not os.path.exists(classPathConfu):
                os.makedirs(classPathConfu)

            # Move the images in confuList to the newly created directory for confusion datapoints
            move(os.path.join(classPath, img), os.path.join(classPathConfu, img))

        # Create separate dir for test datapoints
         for img in testList:
            # Path to the dir containing confusion datapoints of the given class label
            classPathTest = os.path.join(PATHTest, label)

            # Create dirs for storing the test dataset.
            if not os.path.exists(classPathTest):
                os.makedirs(classPathTest)

            # Move the images in TestList to the newly created directory for Test datapoints
            move(os.path.join(classPath, img), os.path.join(classPathTest, img))

        # Create separate dir for train datapoints
         for img in trainList:
            # Path to the dir containing confusion datapoints of the given class label
            classPathTrain = os.path.join(PATHTrain, label)

            # Create dirs for storing the train dataset.
            if not os.path.exists(classPathTrain):
                os.makedirs(classPathTrain)

            # Move the images in TrainList to the newly created directory for Train datapoints
            move(os.path.join(classPath, img), os.path.join(classPathTrain, img))




#  Write your instructions here
# relabelDataSet()
normalizeDataSet()
partitionDataSet(numTrain, numTest, numConfu)
