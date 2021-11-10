import cv2
import numpy as np
import matplotlib as plt
import copy
np.set_printoptions(formatter={'float_kind':"{:0.2f}".format})
# Choose which webcam to capture, 0 for default, 1 for external
#image = cv2.imread('C:\\Users\\profe\\OneDrive\\Skrivebord\\Github\\Group302-P3-Project\\dImages\\testImg.png')
cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

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
    ret, thresh_img = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    #M = cv2.getPerspectiveTransform(approx.astype(np.float32),pts2)
    #print(M)
    #dst = cv2.warpPerspective(frame,M,(300,300))
    c = cv2.waitKey(1)
    if c == 27: #Press escape to exit
        break
    elif c == 32: # Makes it so it only does the contour code when spacebar is clicked
        contours =  cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        for c in contours:
            cv2.drawContours(frame, [c], -1, (255,0,0), 3)
        
            #Approximate contour as a rectangle
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.05 * perimeter, True)
            approx = np.squeeze(approx) #Removes redundant dimension

            # drawing points
            frame2 = copy.copy(frame)
        try:
            #drawing points
            for point in approx:
                x = point[0]
                y = point[1]
                cv2.circle(frame2, (x, y), 3, (0, 255, 0), -1)
                cv2.putText(frame2,str(x),(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1,cv2.LINE_AA)
            
            # drawing skewed rectangle
            cv2.drawContours(frame2, [approx], -1, (0, 255, 0))
            ##print("THIS IS CONTOURS",contours)
            if len(approx) == 4:
                pts2 = np.float32([[0,0],[0,400],[300,400],[300,0]])
        except:
                print("could not find points")
        
        
    # Show the processed webcam feed
    cv2.imshow('Threshold frame', thresh_img)
    cv2.imshow('Camera frame', frame2)

    # Show the processed webcam feed
    cv2.imshow('Threshold frame', thresh_img)
    cv2.imshow('Camera frame', frame)
    #cv2.imshow('Transformed frame', dst)

cap.release()
cv2.destroyAllWindows()