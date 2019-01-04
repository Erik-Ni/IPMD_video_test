
# coding: utf-8

# In[ ]:

import face_recognition
import h5py
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

labels = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


# In[ ]:


def generate_prediction(model, test_X, return_model=False):
    y_prob = model.predict(test_X, batch_size=1, verbose=0)
    if not return_model:
        return y_prob
    else:
        return y_prob, model


# In[ ]:


def loadmodel(path):
    try:
        model = load_model(path)
    except:
        model = model_load(path, img_size=200)
    return model


# In[ ]:


#from IPMD_conversion import conversion
from combine import deep_convert, deep_convert_multi
import glob
import os
import re
import time

model1 = loadmodel('./model%2Fmymodel_0812_Caffe_M200.h5py')


# In[ ]:


from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

video_file = "10005 Sad.mov"

input_movie = cv2.VideoCapture(video_file)
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
fps = input_movie.get(cv2.CAP_PROP_FPS)
width = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))
height =  int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('result_10005 Sad.mp4', fourcc, fps, (width, height))

frequency = -1

# loop over the frames from the video stream
while True:

    frequency += 1
    frequency %= 10
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    c, frame = input_movie.read()
    # Quit when the input video file ends
    if not c:
        break

    if frequency == 0:
        show = 0
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if face_locations:
            show = 10

        # loop over the detections
        for (startY, endX, endY, startX) in face_locations:

            roi = frame[startY:endY, startX:endX]
            roi = cv2.resize(roi, (200,200), interpolation = cv2.INTER_AREA)
            output = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            pixels =  np.asarray([output.flatten()])
            y_prob1 = generate_prediction(model1, pixels)
            y_pred_fst = np.argsort(y_prob1[0])[-1]
            y_pred_snd = np.argsort(y_prob1[0])[-2]
            y_pred_trd = np.argsort(y_prob1[0])[-3]
            
            text_fst = '{}:{:.1%}'.format(labels[y_pred_fst], y_prob1[0][y_pred_fst])
            text_snd = '{}:{:.1%}'.format(labels[y_pred_snd], y_prob1[0][y_pred_snd])
            text_trd = '{}:{:.1%}'.format(labels[y_pred_trd], y_prob1[0][y_pred_trd])

            # draw the bounding box of the face along with the associated
            # probability
            if endX + 30 < width:
                x = endX + 10
            else:
                x = startX - 30
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 255, 0), 2)
            cv2.putText(frame, text_fst, (x, startY+20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, text_snd, (x, startY+50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, text_trd, (x, startY+80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
    elif show > 1:
        for (startY, endX, endY, startX) in face_locations:
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 255, 0), 2)
            cv2.putText(frame, text_fst, (x, startY+20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, text_snd, (x, startY+50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, text_trd, (x, startY+80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    output_movie.write(frame)

# All done!
input_movie.release()
cv2.destroyAllWindows()
