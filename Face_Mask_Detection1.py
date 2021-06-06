import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import sklearn
faceCascade = cv2.CascadeClassifier(r"C:\Users\username\Documents\py\face_mask\haarcascade_frontalface_alt2.xml")
model = load_model(r"C:\Users\username\Documents\py\face_mask\mask_recog.h5")
cap = cv2.VideoCapture(1)
def face_mask_detector(frame):
  # frame = cv2.imread(fileName)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = faceCascade.detectMultiScale(gray, 1.3, 4)
  #print(faces)
  faces_list=[]
  preds=[]
  for (x, y, w, h) in faces:
      face_frame = frame[y:y+h,x:x+w]
      face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
      face_frame = cv2.resize(face_frame, (224, 224))    
      face_frame = img_to_array(face_frame)    
      face_frame = np.expand_dims(face_frame, axis=0)     
      face_frame =  preprocess_input(face_frame)
      faces_list.append(face_frame)
      
      if len(faces_list)>0:
          preds = model.predict(faces_list)
          #print(preds)
      for pred in preds:
          (mask, withoutMask) = pred
          #print(mask)
          #print(withoutMask)
      label = "Mask" if mask > withoutMask else "No Mask"
      #print(label)
      color = (0, 255, 255) if label == "Mask" else (0, 0, 255)
      label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
      cv2.putText(frame, label, (x, y- 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

      cv2.rectangle(frame, (x, y), (x + w, y + h),color, 1)
   #cv2.imshow('img',frame)
  #print(faces_list)
  #print(len(faces_list))
  return frame



print("Processing Video...")
while True:
  _, frame = cap.read()
  output = face_mask_detector(frame)
  cv2.imshow('img',frame)
  k = cv2.waitKey(10)&0xff
  if  k==27:
      break
cap.release()




