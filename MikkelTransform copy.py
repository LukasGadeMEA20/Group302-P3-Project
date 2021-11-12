import cv2
import numpy as np
import matplotlib as plt
import copy
np.set_printoptions(formatter={'float_kind':"{:0.2f}".format})

image = cv2.imread('C:\\Users\\profe\\Documents\\GitHub\\Group302-P3-Project\\dImages\\testImg.png')
frame = image

def grayScale(frame):
    # Our operations on the frame come here
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Converting the current frame to gray

def blurImage(gray):
    return cv2.GaussianBlur(gray,(5,5),0) #Blurring the grey frame

def threshholdImage(gray):
    return cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

def findContours(thresh_img):
    return cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

def drawContours(contours, frame, copiedFrame):
    for c in contours:
        cv2.drawContours(frame, [c], -1, (255,0,0), 3)
    
        #Approximate contour as a rectangle
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * perimeter, True)
        approx = np.squeeze(approx) #Removes redundant dimension

        # drawing points
        try:
            #drawing points
            for point in approx:
                x = point[0]
                y = point[1]
                cv2.circle(copiedFrame, (x, y), 3, (0, 255, 0), -1)
                cv2.putText(copiedFrame,str(x),(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1,cv2.LINE_AA)
        
            # drawing skewed rectangle
            cv2.drawContours(copiedFrame, [approx], -1, (0, 255, 0))
            ##print("THIS IS CONTOURS",contours)
            if len(approx) == 4:
                pts2 = np.float32([[0,0],[0,400],[300,400],[300,0]])
                M = cv2.getPerspectiveTransform(approx.astype(np.float32),pts2)
                #print(M)
                dst = cv2.warpPerspective(frame,M,(300,400))
                cv2.imshow('Transformed frame', dst)

                #Crop the image to get the artwork
                croppedImg = dst[50:220, 30:270]
                #Show cropped image
                cv2.imshow("Cropped image", croppedImg)
        except:
            print("could not find points")
    #elif c == 32: # Makes it so it only does the contour code when spacebar is clicked
        
        
if __name__ == '__main__':
    while True:
        copiedFrame = copy.copy(frame)
        gray = grayScale(frame)
        ret, thresh_img = threshholdImage(gray)
        contours = findContours(thresh_img)
        drawContours(contours,frame,copiedFrame)

        
        # Show the processed webcam feed
        cv2.imshow('Threshold frame', thresh_img)
        cv2.imshow('Camera frame', copiedFrame)

        # Show the processed webcam feed
        cv2.imshow('Threshold frame', thresh_img)
        cv2.imshow('Camera frame', frame)
        #cv2.imshow('Transformed frame', dst)

        c = cv2.waitKey(1)
        if c == 27: #Press escape to exit
            break

    cv2.destroyAllWindows()
