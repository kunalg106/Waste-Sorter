import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from skimage.transform import resize
from random import shuffle


class Dataset(object):
    # This class contains all the methods which maybe required for
    # creating a dataset. Its attributes and several methods are private

    def __init__(self, PATH, num_train, img_size):

        self.__PATH = PATH
        self.__img_size = img_size
        self.__train_index = 0 # index till which training data have been read

        dataset_list = self.__get_dataset_list()
        self.__train_list, self.__test_list = self.__prepare_dataset(dataset_list, num_train)
        # self.__X_test, self.__Y_test = self.__read_img_from_list(self.__test_list)
        self.__read_img_from_list(self.__test_list, 'test')
        pass


    def __get_dataset_list(self):
        # This method searches the entire dir provided in the path and creates
        # a complete list containing all the complete dataset

        # Create labels corresponding to the several sub-dir names.
        label_names = [label for label in os.listdir(self.__PATH) if label[0] != '.']

        # label_code = {key:label for label,key in enumerate(label_names)}


        dataset_list = list() # Complete list stored here
        #  The list is stored in the format (file name, file label) where
        #  file label is the category which the file belongs to
        max_imgs = 1000
        idx = 0

        for label in label_names:
            data_class_path = os.path.join(self.__PATH, label) # create path to sub-dir
            # Append all the files in sub-dir to the main list
            # print("Loading Image from %s", label)
            idx = 0
            # print(label)
            for img in os.listdir(data_class_path):
                if(idx > max_imgs):
                    break
                else:
                    idx +=1
                    progress(idx, max_imgs, status= 'Loading Images from ' + label)
                if (img[0] != '.'):
                    dataset_list.append((img,label))
        return dataset_list # return the complete list

    def __prepare_dataset(self,dataset_list, num_train):
        print("Preparing dataset")
        # This method separates the complete data list into respective
        #  lists of training anf testing data

        # Firstly the entire list is shuffled to remove any bais
        shuffle(dataset_list)   # check is the data set is normalized between train and test
        train_list = dataset_list[:num_train]
        test_list = dataset_list[num_train:]

        return train_list, test_list # Return training and testing lists

    def __read_img_from_list(self, dataset_list, type):
        print("Reading images from list")
        #  This method reads the image from the disk as per the provided list
        #  Try making a csv file
        # Get the labels
        label_names = [label for label in os.listdir(self.__PATH) if label[0] != '.']

        # Define a mapping between string labels and corresponding integral symbols
        label_code = {key:label for label,key in enumerate(label_names)}
        # print(label_code)
        # One hot code for image classification using softmax
        one_hot_code = {'cardboard': [1,0,0,0,0,0], \
                        'plastic'  : [0,1,0,0,0,0], \
                        'metal'    : [0,0,1,0,0,0], \
                        'paper'    : [0,0,0,1,0,0], \
                        'trash'    : [0,0,0,0,1,0], \
                        'glass'    : [0,0,0,0,0,1]}
        # print(one_hot_code)
        # The three dimensional images are converted into a 1-D array of size img_len
        img_len  = self.__img_size[0]*self.__img_size[1]*self.__img_size[2]

        # Initialize the data and label arrays
        # X = np.zeros((len(dataset_list), img_len)) # data
        X = np.zeros((len(dataset_list), self.__img_size[0], self.__img_size[1], self.__img_size[2]))
        Y = np.zeros((len(dataset_list),6)) #label

        # Read each image (and corresponding label) in the dataset_list
        #  Add them to the respective data and label arrays
        idx = 0
        for name, label in dataset_list:
            img_path = os.path.join(self.__PATH, label, name) # path to image file
            # Read and resize the image
            # X[idx,:] = resize(misc.imread(img_path), self.__img_size).reshape(img_len,)
            X[idx,:] = resize(misc.imread(img_path), self.__img_size)
            Y[idx,:] = one_hot_code[label]
            idx +=1

        if(type == 'test'):
            self.__X_test = self.__preprocess_dataset(X, 'test')
            self.__Y_test = Y
        elif(type == 'train'):
            self.__X_train = self.__preprocess_dataset(X, 'train')
            self.__Y_train = Y
        else:
            print("Invalid dataset type")

        return # return the data and label arrays

    def __preprocess_dataset(self, arrImg, type):

        if(type == 'test'):
            # Calculate the mean Image.
            self.__meanTestImg = np.mean(arrImg,axis = 0)
            # Subtract the mean Image
            arrImg -= self.__meanTestImg

            # Calculate the Norm for each pixel across all the images
            self.__arrNormTest = np.linalg.norm(arrImg, axis = 0)
            # Normalize the image array using the norm array
            arrImg = np.divide(arrImg, self.__arrNormTest)

        elif(type == 'train'):
            # Calculate the mean Image.
            self.__meanTrainImg = np.mean(arrImg,axis = 0)
            # Subtract the mean Image
            arrImg -= self.__meanTrainImg

            # Calculate the Norm for each pixel across all the images
            self.__arrNormTrain = np.linalg.norm(arrImg, axis = 0)
            # Normalize the image array using the norm array
            arrImg = np.divide(arrImg, self.__arrNormTrain)

        else:
            print("Invalid dataset type")


        # print(arrNorm.shape)

        return arrImg

    def get_next_batch(self, batch_size):
        # This method reads the next batch of trainig images from the
        # train list and returns the corresponding data and label arrays
        self.__read_img_from_list(self.__train_list[self.__train_index:self.__train_index + batch_size], 'train')
        self.__train_index += batch_size # update the index for later training data
        # print(self.__X_train.shape, self.__Y_train.shape)
        return self.__X_train, self.__Y_train # return the training data and label arrays

    def is_end_of_list(self,batch_size):
        if(self.__train_index + batch_size > len(self.__train_list)):
            return 1
        else:
            return 0

    def reset_train_index(self):
        self.__train_index = 0


    def get_test_data(self):
        # This method returns the test data and label arrays
        return self.__X_test, self.__Y_test

    def print_image(self, type, idx):
        if(type == 'test'):
            print('test')
            X_test = np.multiply(self.__X_test, self.__arrNormTest) + self.__meanTestImg
            plt.imshow(X_test[idx,:])
            plt.show()
        elif(type == 'train'):
            print('train')
            X_train = np.multiply(self.__X_train, self.__arrNormTrain) + self.__meanTrainImg
            print(X_train)
            plt.imshow(X_train[idx,:])
            plt.show()
        else:
            print("Input correct image dataset type")




if __name__ == '__main__':

    PATH = 'dataset-resized' # path to the dir containing sub-dirs of images
    img_size = (128,128,3) # Normalized image size to be used for learning
    num_train = 2520 # number of training datapoints
    batch_size = 10
    data_set = Dataset(PATH, num_train, img_size)
    X_test, Y_test = data_set.get_test_data()

    print('#######################################')
    print(X_test.shape, Y_test.shape)
    print('#######################################')

    X_train, Y_train = data_set.get_next_batch(batch_size)
    print(X_train.shape, Y_train.shape)
    X_train, Y_train = data_set.get_next_batch(batch_size)
    print(X_train.shape, Y_train.shape)
    data_set.print_image('test', 1)
    # print(X_test[1,:])
    # for i in range(10):
    #     X_train, Y_train = data_set.get_next_batch(batch_size)
    #     print(Y_train)
    #     plt.imshow(X_train[0,:].reshape(img_size))
    #     plt.show()
def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush() # As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)
