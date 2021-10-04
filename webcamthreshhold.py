import cv2
import numpy as np

# Choose which webcam to capture, 0 for default, 1 for external
cap = cv2.VideoCapture(1)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    # Capture frame by frame
    ret, frame = cap.read()

    # Resizing the webcam display size
    frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_AREA)

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Converting the current frame to gray
    blur = cv2.GaussianBlur(gray,(5,5),0) #Blurring the grey frame
    ret, thresh_img = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)

    contours =  cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    for c in contours:
        #cv2.drawContours(frame, [c], -1, (255,0,0), 3)
    
        #Approximate contour as a rectangle
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * perimeter, True)

        # drawing points
        for point in approx:
            x, y = point[0]
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        # drawing skewed rectangle
        cv2.drawContours(frame, [approx], -1, (0, 255, 0))

    # Show the processed webcam feed
    cv2.imshow('Input', frame)

    c = cv2.waitKey(1)
    if c == 27: #Press escape to exit
        break

cap.release()
cv2.destroyAllWindows()