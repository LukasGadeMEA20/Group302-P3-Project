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

    gray = np.float32(frame2)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    dst = cv2.dilate(dst, None)

    frame[dst>0.01*dst.max()]=[0,0,255]

    #retval, img_binary = cv2.threshold(frame2,127, 255, cv2.THRESH_BINARY)
    #img_binary = 255.0*(frame2/255.0)**2
    cv2.imshow('dst', frame)

    c = cv2.waitKey(1)
    if c == 27: #Press escape to exit
        break

cap.release()
cv2.destroyAllWindows()