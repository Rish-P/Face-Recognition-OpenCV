import cv2
import matplotlib.pyplot as plt
import numpy as np

face_cascade = cv2.CascadeClassifier(
    'cascades/data/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # command extracts out the faces in the frame that have been detected
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.5, minNeighbors=5)

    # looping over the faces recognized by the algorithm, (x,y,w,h) are the coordinates of the face detected
    for (x, y, w, h) in faces:
        width = x + w
        height = y + h
        # the region of interest i.e portion of the frame where the face has been detected
        roi_frame = frame[x:x+h, y:y+h]
        # drawing rectangle box around the face
        cv2.rectangle(frame, (x, y), (width, height), (0, 0, 255), thickness=3)

    cv2.imshow("main frame", frame)

    if cv2.waitKey(0) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
