# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 22:37:51 2021

@author: edibe
"""

import cv2 
import pickle
import numpy as np 

def Preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    
    return img

cap = cv2.VideoCapture(0)
cap.set(3, 480)
cap.set(4, 480)

pickle_in = open("model_trained_new.p", "rb")
model = pickle.load(pickle_in)

while True:
    success, frame = cap.read()
    img = np.asarray(frame)
    img = cv2.resize(img, (32, 32))
    img = Preprocess(img)
    
    img = img.reshape(1, 32, 32, 1)
    
    #predict 
    classIndex = int(model.predict_classes(img))
    predictions = model.predict(img)
    probVal = np.amax(predictions)
    print(classIndex, probVal)
    
    if classIndex == 0:
            className = "Person"
    elif classIndex == 1:
            className = "Cat"
            
    if probVal > 0.7:
        cv2.putText(frame, str(className)+ "    " + str(probVal), (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))
        
    cv2.imshow("Person OR Cat", frame)
    
    if cv2.waitKey(1) & 0XFF == ord("q"): break