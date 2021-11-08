from cv2 import imshow


import cv2
import numpy as np

# Preprocesses the image and returns the thresholded image
def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Converting the current frame to gray
    retval, thresh_img = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    return thresh_img

def find_contours(frame, thresh_img):
    contours =  cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    for c in contours:
        cv2.drawContours(frame, [c], -1, (255,0,0), 3)
    
        #Approximate contour as a rectangle
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * perimeter, True)
        approx = np.squeeze(approx) #Removes redundant dimension

       # drawing points
        for point in approx:
            x = point[0]
            y = point[1]
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            cv2.putText(frame,str(x),(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1,cv2.LINE_AA)

        # drawing skewed rectangle
        cv2.drawContours(frame, [approx], -1, (0, 255, 0))