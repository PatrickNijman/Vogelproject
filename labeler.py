# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 11:58:21 2020

@author: Pat
"""

from datetime import datetime
import numpy as np
import imutils
import cv2
from birdwatch import DBmanager
from threading import Thread
import os
from keras.models import load_model

# model = load_model('my_model_fc__birds.hdf5')
birds = []
dates = []
bird_path = 'C:/Users/Pat/.spyder-py3/ImageData/training_set/Birds/'
notbird_path = 'C:/Users/Pat/.spyder-py3/ImageData/training_set/NotBirds/'
if not os.path.exists(r'ImageData/training_set'):
    os.makedirs('ImageData/training_set')
    
if not os.path.exists(r'ImageData/training_set/Birds'):
    os.makedirs('ImageData/training_set/Birds')
    
if not os.path.exists(r'ImageData/training_set/NotBirds'):
    os.makedirs('ImageData/training_set/NotBirds')
  
DB = DBmanager()
while True:
    img, dt = DB.pop_image()
    if img is None:
        break
    
    cv2.imshow('frame', img)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()
    im2 = cv2.resize(img, (224, 224))/255
    im2 = im2.reshape((1,) + im2.shape)
    # pr = model.predict(im2)
    # print2(pr, dt)#
    q = int(input('Bird[0], no bird[1], stop[9]'))

    if q == 0:
        cv2.imwrite(os.path.join(bird_path , '{}.jpg'.format(len(os.listdir(bird_path))+1)), img)
    elif q == 1:
        cv2.imwrite(os.path.join(notbird_path , '{}.jpg'.format(len(os.listdir(notbird_path))+1)), img)
    elif q==9:
        break
    else:
        pass

DB._send_data([], [], birds, dates)