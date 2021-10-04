import cv2
import numpy as np

cap = cv2.VideoCapture(1)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
    frame2 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    retval, img_binary = cv2.threshold(frame2,127, 255, cv2.THRESH_TOZERO)

    cv2.imshow('Input', img_binary)

    c = cv2.waitKey(1)
    if c == 27: #Press escape to exit
        break