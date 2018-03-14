import os

PATH = 'dataset'

label_names = [label for label in os.listdir(PATH) if label[0] != '.']

for label in label_names:
    data_class_path = os.path.join(PATH, label) # create path to sub-dir
    # Append all the files in sub-dir to the main list
    i = 1
    # print(os.listdir(data_class_path))
    # print('#################')
    for img in os.listdir(data_class_path):
        if (img[0] != '.'):
            os.rename(data_class_path + '/' + img,data_class_path + '/' + label + str(i))
            i+= 1
