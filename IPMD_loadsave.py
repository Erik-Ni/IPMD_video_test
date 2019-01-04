# load model by building architecture and load weight


import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import pandas as pd
import pickle
import cv2
from Utils import load_pkl_data
from Utils import load_pd_data
from Utils import load_pd_direct
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten,Dropout
from tensorflow.python.keras.models import load_model
#from keras.models import load_model
#from keras.models import model_from_json
from tensorflow.python.keras.optimizers import Adam


def build_cnn_model(img_size, num_classes):
    #img_size = 48
    img_size_flat = img_size * img_size
    img_shape = (img_size, img_size, 1)
    #num_channels = 1
    #num_classes = 8
    # Start construction of the Keras.
    model = Sequential()
    model.add(InputLayer(input_shape=(img_size_flat,)))
    #model.add(input_shape=(img_size_flat,))
    model.add(Reshape(img_shape))

    #model.add(Dropout(0.5, input_shape=(48, 48, 1)))
    model.add(Conv2D(kernel_size=5, strides=1, filters=32, padding='same',
                     activation='relu'))
    model.add(Conv2D(kernel_size=5, strides=1, filters=32, padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Conv2D(kernel_size=10, strides=1, filters=64, padding='same',
                     activation='relu'))
    model.add(Conv2D(kernel_size=10, strides=1, filters=64, padding='same',
                     activation='relu'))
    model.add(Conv2D(kernel_size=10, strides=1, filters=64, padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Conv2D(kernel_size=15, strides=1, filters=128, padding='same',
                     activation='relu'))
    model.add(Conv2D(kernel_size=15, strides=1, filters=128, padding='same',
                     activation='relu'))
    model.add(Conv2D(kernel_size=15, strides=1, filters=128, padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    
    model.add(Conv2D(kernel_size=20, strides=1, filters=256, padding='same',
                     activation='relu'))
    model.add(Conv2D(kernel_size=20, strides=1, filters=256, padding='same',
                     activation='relu'))
    model.add(Conv2D(kernel_size=20, strides=1, filters=256, padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))


    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    # Last fully-connected / dense layer with softmax-activation
    # for use in classification.
    model.add(Dense(num_classes, activation='softmax'))

    return model


def compile_model(model, optimizer, loss='categorical_crossentropy', metrics=['accuracy']):
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    return model

def model_save(model, file_path):
    model.save_weights(file_path)

def model_load(file_path,img_size=200, num_classes=8):
    model = build_cnn_model(img_size, num_classes)
    optimizer = Adam(lr = 1e-4)
    model = compile_model(model, optimizer)
    model.load_weights(file_path)
    return model