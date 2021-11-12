import cv2
import numpy as np
import matplotlib as plt
import copy
import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

np.set_printoptions(formatter={'float_kind':"{:0.2f}".format})
# Choose which webcam to capture, 0 for default, 1 for external
#image = cv2.imread('C:\\Users\\profe\\OneDrive\\Skrivebord\\Github\\Group302-P3-Project\\dImages\\testImg.png')
def loadCamera():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    return cap

def setCameraSize(cap):
    # Capture frame by frame
    ret, frame = cap.read()
    
    # Capture frame by frame
    return cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_AREA)
    
#while True:
    # Capture frame by frame
    #ret, frame = cap.read()

    # Resizing the webcam display size
    #frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_AREA)

def grayScale(frame):
    # Our operations on the frame come here
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Converting the current frame to gray

def blurImage(gray):
    return cv2.GaussianBlur(gray,(5,5),0) #Blurring the grey frame

def threshholdImage(gray, tVal):
    return cv2.threshold(gray, tVal, 255, cv2.THRESH_BINARY_INV)

def findContours(thresh_img):
    return cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

def loadCardArtwork(filePath):
    cardArtwork = []
    i = 0

def preProcData():
    train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.2, horizontal_flip=True)
    training_set = train_datagen.flow_from_directory('../cardArtwork', color_mode='rgb')

def defData():
    cnn = tf.keras.models.Sequential()
    cnn.add(tf.keras.)

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

                #Crop the image to get the artwork and show it
                croppedImg = dst[50:220, 30:270]
                cv2.imshow("Cropped image", croppedImg)

                #Preprocess the cropped image and show it
                greyCrop = grayScale(croppedImg)
                ret, threshCrop = threshholdImage(greyCrop, 80)
                cv2.imshow("Hej", threshCrop)
        except:
            print(" ")
    #elif c == 32: # Makes it so it only does the contour code when spacebar is clicked
        
        
if __name__ == '__main__':
    cap = loadCamera()
    while True:
        frame = setCameraSize(cap)
        copiedFrame = copy.copy(frame)
        gray = grayScale(frame)
        ret, thresh_img = threshholdImage(gray, 100)
        contours = findContours(thresh_img)
        drawContours(contours, frame, copiedFrame)

        
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

    cap.release()
    cv2.destroyAllWindows()


