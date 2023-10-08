#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Load required packages
import scipy.io as sio
import numpy as np
import tensorflow as tf
#from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
import numpy.ma as ma
from PIL import Image                                                            
import glob
import os
import rospy
import random
from geometry_msgs.msg import Twist



os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# List of arrow classes
namesList = ['up', 'down', 'left', 'right']


# Add here the code to load the testing database
imageFolderTestingPath = r'/rosdata/ros_ws_loc/src/differential_robot/src/scripts/validation'
imageTestingPath = []

for i in range(len(namesList)):
    testingLoad = imageFolderTestingPath + '/' + namesList[i] + '/*.jpg'
    imageTestingPath = imageTestingPath + glob.glob(testingLoad)
    
updateImageSize = [128, 128]
tempImg = Image.open(imageTestingPath[0]).convert('L')
tempImg.thumbnail(updateImageSize, Image.ANTIALIAS)
[imWidth, imHeight] = tempImg.size

# create space to load testing images
x_test = np.zeros((len(imageTestingPath), imHeight, imWidth, 1))

# load testing images
for i in range(len(x_test)):
    tempImg = Image.open(imageTestingPath[i]).convert('L')
    tempImg.thumbnail(updateImageSize, Image.ANTIALIAS)
    x_test[i, :, :, 0] = np.array(tempImg, 'f')

#model = load_model(r'trained_arrows.h5')
model = load_model(r'/rosdata/ros_ws_loc/src/differential_robot/src/scripts/trained_arrows.h5')
# Move robot
rospy.init_node("arrow")
pub = rospy.Publisher("/cmd_vel", Twist,  queue_size = 1)
msg = Twist()

    
while not rospy.is_shutdown():
    
    test_image = (random.choice(x_test))
    test_image = test_image.reshape((1, 96, 128, 1))
    
    plt.imshow(test_image.squeeze())
    plt.show()
    

    output = model.predict_on_batch(test_image)
    direction = namesList[np.argmax(output)]
    
    if direction == "up":
        msg.linear.x = 0.2
        msg.angular.z = 0
        pub.publish(msg)
    if direction == "down":
        msg.linear.x = -0.2
        msg.angular.z = 0
        pub.publish(msg)
    if direction == "left":
        msg.linear.x = 0
        msg.angular.z = -0.4
        pub.publish(msg)
    if direction == "right":
        msg.linear.x = 0
        msg.angular.z = 0.4
        pub.publish(msg)
        
    rospy.loginfo("Direction: %s", direction)
    
    input("Press 'Enter' for new image: ")
        
    pass
    