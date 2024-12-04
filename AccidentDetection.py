from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import cv2
import random
import numpy as np
from keras.layers import Activation
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import winsound
from keras.models import model_from_json
import pickle



global filename
global classifier



names = ['Accident Occured','No Accident Occured']



def detect():

    print("detecting") 
    video_capture = cv2.VideoCapture(0)

    wait=0
    with open('model1.json', "r") as json_file:
            print("loading model")
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
    classifier.load_weights("model/model_weights1.h5")
     
    print(classifier.summary())
    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        key=cv2.waitKey(100)
        wait+=100

        result=""


        if key==ord('q'):
            break
        if wait==5000:
            img = frame
            img = cv2.resize(img, (120,120))
            img = img.reshape(1, 120,120, 3)
            filename="frame"+".jpg"
            cv2.imwrite(filename,frame)
            wait=0
            predict = classifier.predict(img)
            print(np.argmax(predict))
            result += names[np.argmax(predict)]
        print(result)
        cv2.putText(frame, result, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
            #cv2.imshow("video frame", frame)
        

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

detect()