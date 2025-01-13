import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer
import time

model = load_model("ModelE5_85.h5", compile=False)

mixer.init()
sound = mixer.Sound('alarm.wav')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")



lbl=['Close', 'Open']

path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
counter = 0

while(True):
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,minNeighbors = 1,scaleFactor = 1.1,minSize=(25,25))
    eyes = eye_cascade.detectMultiScale(gray,minNeighbors = 1,scaleFactor = 1.1)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (255,0,0) , 3 )

    for (x,y,w,h) in eyes:

        eye = frame[y:y+h, x:x+w]
        eye = cv2.resize(eye, (224, 224))  # Resize without converting to grayscale
        
        # Ensure the image is in the correct format (RGB)
        eye = cv2.cvtColor(eye, cv2.COLOR_BGR2RGB)
        
        # Convert to a NumPy array
        eye = np.array(eye)
        
        # # Normalize pixel values (if the model requires it, typical for pre-trained models)
        # eye = eye / 255.0
        
        # Convert to tensor and add a batch dimension
        eye = tf.expand_dims(eye, axis=0)  # Shape: (1, 224, 224, 3)
        
        print(eye.shape)  # Debug: Should print (1, 224, 224, 3)
        
        # Predict
        prediction = model.predict(eye)
        print(prediction)

       #Condition for Close
        if prediction[0][0]<=0.50:
            cv2.putText(frame,"Closed"+str(counter),(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(frame,'Drowsy',(100,height-20), font, 1,(0,0,255),1,cv2.LINE_AA)
            counter=counter+1
            print("Close Eyes")
            if(counter >= 4):
                try:
                    sound.play()
                except:  # isplaying = False
                    pass

        #Condition for Open
        else:
            counter = 0
            cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
            print("Open Eyes")
            cv2.putText(frame,'Awake'+str(counter),(100,height-20), font, 1,(0,255,0),1,cv2.LINE_AA)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
