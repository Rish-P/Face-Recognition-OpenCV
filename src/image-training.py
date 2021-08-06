import os
from PIL import Image
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #gets the path in which present file is in
image_dir = os.path.join(BASE_DIR, "images") #appends 'images' at the end of root path

for root,dirs,files in os.walk(image_dir): #walk through each folder/file in the image directory
    for file in files:
        if file.endswith("png"):
            path = os.path.join(root, file)   #full path of the image file appended from the root
            label = os.path.basename(root).lower() #getting the label name, which will be used for face detection
            #print(label)
            img = Image.open(path).convert("L") #opening the image in grayscale
            img_array = np.array(img, np.uint8) #convert the image into the array format

            #importing the viola jones algo
            face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

            faces = face_cascade.detectMultiScale(img_array, scaleFactor=1.5, minNeighbors=2)

            for face in faces:
                roi_x = face
