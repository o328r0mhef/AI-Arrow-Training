# -*- coding: utf-8 -*-
"""
Detection of arrows for robot control using CNN and ROS

"""

# Load required packages
from PIL import Image  
import scipy.io as sio
import numpy as np
import glob
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
import numpy.ma as ma
                                                          


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

np.random.seed(7)
    

# List of arrow classes
namesList = ['up', 'down', 'left', 'right']

imageFolderTrainingPath = r'train'
imageFolderTestingPath = r'validation'
imageTrainingPath = []
imageTestingPath = []


# full path to training and testing images
for i in range(len(namesList)):
    trainingLoad = imageFolderTrainingPath + '/' + namesList[i] + '/*.jpg'
    testingLoad = imageFolderTestingPath + '/' + namesList[i] + '/*.jpg'
    imageTrainingPath = imageTrainingPath + glob.glob(trainingLoad)
    imageTestingPath = imageTestingPath + glob.glob(testingLoad)
    
# print number of images for training and testing
print(len(imageTrainingPath))
print(len(imageTestingPath))

# resize images to speed up training process
updateImageSize = [128, 128]
tempImg = Image.open(imageTrainingPath[0]).convert('L')
#tempImg = Image.open(imageTrainingPath).convert('L')
tempImg.thumbnail(updateImageSize, Image.ANTIALIAS)
[imWidth, imHeight] = tempImg.size

# create space to load training images
x_train = np.zeros((len(imageTrainingPath), imHeight, imWidth, 1)) 
# create space to load testing images
x_test = np.zeros((len(imageTestingPath), imHeight, imWidth, 1))


# load training images
for i in range(len(x_train)):
    tempImg = Image.open(imageTrainingPath[i]).convert('L')
    tempImg.thumbnail(updateImageSize, Image.ANTIALIAS)
    x_train[i, :, :, 0] = np.array(tempImg, 'f')
    
# load testing images
for i in range(len(x_test)):
    tempImg = Image.open(imageTestingPath[i]).convert('L')
    tempImg.thumbnail(updateImageSize, Image.ANTIALIAS)
    x_test[i, :, :, 0] = np.array(tempImg, 'f')


# create space to load training labels
y_train = np.zeros((len(x_train),));
# create space to load testing labels
y_test = np.zeros((len(x_test),));


# load training labels
countPos = 0
for i in range(0, len(namesList)):
    for j in range(0, round(len(imageTrainingPath)/len(namesList))):
        y_train[countPos,] = i
        countPos = countPos + 1
    
# load testing labels
countPos = 0
for i in range(0, len(namesList)):
    for j in range(0, round(len(imageTestingPath)/len(namesList))):
        y_test[countPos,] = i
        countPos = countPos + 1
        
# convert training labels to one-hot format
y_train = tf.keras.utils.to_categorical(y_train, len(namesList));
# convert testing labels to one-hot format
y_test = tf.keras.utils.to_categorical(y_test, len(namesList));
        


# CNN model created, composed of convolution, maxpooling and fully connected layers.
model = Sequential()
# First feature layer composed of a convolution and maxpooling layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(96, 128, 1)))
model.add(layers.MaxPooling2D((2, 2)))
# Second feature layer composed of a convolution and maxpooling layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the information from the previous layer
model.add(layers.Flatten())
# Fully connected layer with 64 neurons and relu activation
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))


# Compile, fit and evaluate CNN model.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Trains the model with a define number of epochs, batch size and validation
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
# Evaluate the performance of the train model using the test dataset
score = model.evaluate(x_test, y_test, batch_size=5)

# Saves the trained model
idFile = 'trained_arrows';
modelPath = idFile +'.h5'
model.save(modelPath);


# Displays the recognition accuracy
print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

# plot accuracy curves
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# plot loss curves
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Plot accuracy plots

print('OK')