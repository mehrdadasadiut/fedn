#!./.mnist-keras/bin/python
#########You should download the training set from https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip and unzip / put in in bin directory################
import numpy as np
import fire
import cv2 as cv
import os
from sklearn.model_selection import train_test_split
#get_data(out_dir='data'):
classes = 43
images = []
labels = []
out_dir='data'
dataset = 'bin/Final_Training/Images'
for i in range(classes):
    path = os.path.join(dataset, str(str(i).zfill(5)))
    img_folder = os.listdir(path)
    for j in img_folder:
        try:
            image = cv.imread(str(path+'/'+j))
            image = cv.resize(image, (32, 32))
           #image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            image = np.array(image)
            images.append(image)
            label = np.zeros(classes)
            label[i] = 1.0
            labels.append(label)
        except:
            pass
images = np.array(images)
#images = images/255
labels = np.array(labels)
print('Images shape:', images.shape)
print('Labels shape:', labels.shape)

##SPLITING THE DATASET
#X = images.astype(np.float32)
#y = labels.astype(np.float32)
X = images
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

##Saving to npz
np.savez(f'{out_dir}/traffic.npz', x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test)
#if __name__ == '__main__':
#    fire.Fire(get_data)
