
# coding: utf-8

# In[1]:


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
import matplotlib.patches as patches
import imutils
from tensorflow.python.keras.models import load_model
#from keras.models import load_model
#from keras.models import model_from_json
from tensorflow.python.keras.optimizers import Adam
from IPMD_loadsave import model_save, model_load


# In[2]:


labels = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'] 


# In[3]:


def generate_prediction(model, test_X, return_model=False):    
    y_prob = model.predict(test_X, batch_size=1, verbose=0)
    if not return_model:
        return y_prob
    else:
        return y_prob, model


# In[4]:


def loadmodel(path):
    try:
        model = load_model(path)
    except:
        model = model_load(path, img_size=200)
    return model


# In[5]:


#from IPMD_conversion import conversion
from combine import deep_convert, deep_convert_multi
import glob
import os
import re
import time

model1 = loadmodel('./model%2Fmymodel_0812_Caffe_M200.h5py')


# In[6]:


from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2


# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

frequency = 29

# loop over the frames from the video stream
while True:
    
    frequency += 1
    frequency %= 30
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
 
    if frequency == 0:
        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < 0.5:
                continue
                
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            roi = frame[startY:endY, startX:endX]
            roi = cv2.resize(roi, (200,200), interpolation = cv2.INTER_AREA)
            output = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            pixels =  np.asarray([output.flatten()])
            y_prob1 = generate_prediction(model1, pixels)
            y_pred = np.argmax(y_prob1)

            text = '{}:{:.1%}'.format(labels[y_pred], y_prob1[0][y_pred])


            # draw the bounding box of the face along with the associated
            # probability
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    else:
        for i in range(0, detections.shape[2]):
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        
    # show the output frame
    cv2.imshow("Frame", frame)

    # if the `q` key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

