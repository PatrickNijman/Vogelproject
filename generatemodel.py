# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 09:53:07 2020

@author: Pat
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import random
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils import *
from keras.callbacks import *
from keras.initializers import *
from keras.metrics import categorical_crossentropy
from keras.preprocessing import image
from keras.applications import imagenet_utils
import matplotlib.pyplot as plt
from PIL import ImageFile
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import keras
import tensorflow as tf
import keras.backend as K
tf.keras.backend.clear_session()


ep = 30
batch_size = 32
learning_rate = 0.0001
classes = ['House sparrow', 'Great tit', 'Eurasian blue tit', 'Eurasian magpie', 'Eurasian jay']
seed = random.randint(1, 1000)

#Dit uncommenten als je met GPU wilt trainen

# tf.debugging.set_log_device_placement(True) 
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


a = 224
b = 224
ImageFile.LOAD_TRUNCATED_IMAGES = True
base_model=MobileNetV2(include_top=False, weights='imagenet', input_shape=(a,b, 3))#(216, 384, 3)
base_model.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
# x = Dropout(rate = .2)(x)
x = BatchNormalization()(x)
x = Dense(1280, activation='relu',  kernel_initializer=glorot_uniform(seed), bias_initializer='zeros')(x)
# x = Dropout(rate = .2)(x)
x = BatchNormalization()(x)
predictions = Dense(len(classes), activation='softmax', kernel_initializer='random_uniform', bias_initializer='zeros')(x)

model = Model(inputs=base_model.input, outputs=predictions)


optimizer = Adam(lr=learning_rate)
# optimizer = RMSprop(lr=learning_rate)

loss = "categorical_crossentropy"
# loss = "kullback_leibler_divergence"

# for layer in model.layers:
#     layer.trainable = True


model.compile(optimizer=optimizer,
              loss=loss,
              metrics=["accuracy"])


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = False)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('ImageData/training_set',
                                                  target_size = (a,b), 
                                                  color_mode='rgb',
                                                 batch_size = batch_size , 
                                                 class_mode = 'categorical',
                                                 shuffle=True)

test_set = test_datagen.flow_from_directory('ImageData/test_set', 
                                             target_size = (a,b), 
                                             color_mode='rgb',
                                            batch_size = batch_size , 
                                            class_mode = 'categorical')

step_size_train=int(training_set.n//training_set.batch_size)

history = model.fit_generator(training_set, steps_per_epoch= step_size_train, epochs = ep, validation_data = test_set)

model.save('my_model_fc__birds.hdf5')


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
# test_set_2 = test_datagen.flow_from_directory('ImageData/test_set_2',                                       
#                                               target_size = (216, 384), 
#                                               batch_size = 32,
#                                               class_mode = 'binary')

# scores = classifier.evaluate_generator(test_set_2,500) #500 testing images
# print("Accuracy = ", scores[1])