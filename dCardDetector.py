import cv2
import numpy as np
import dCards

# Choose which webcam to capture, 0 for default, 1 for external
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    # Capture frame by frame
    ret, frame = cap.read()

    # Resizing the webcam display size
    frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_AREA)

    #Preprocessing the video input (from dCards.py)
    preProc = dCards.preprocess_image(frame)

    #Draw contours
    contours =  cv2.findContours(preProc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    for c in contours:
        cv2.drawContours(frame, [c], -1, (255,0,0), 3)

    #Show video stream
    cv2.imshow('Input', frame)

    c = cv2.waitKey(1)
    if c == 27: #Press escape to exit
        break

cap.release()
cv2.destroyAllWindows()

