# Importing libs
import cv2
import numpy as np
import dannyfunctions

# Webcam capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)    #Get video input
if not cap.isOpened():                      #Checks if the webcam can be found
    raise IOError("Cannot open webcam")

# While the program is running
while True:
    # Capture frame by frame
    frame = cap.read()

    # Resizing the webcam display size
    frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_AREA)

    pre