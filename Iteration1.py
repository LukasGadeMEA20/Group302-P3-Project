import cv2
import numpy as np
import matplotlib as plt
import copy
import os
import pyautogui

DATABASE_PATH = 'card_data_base'
N_IMAGES = 6
IMG_WIDTH = 240
IMG_HEIGHT = 170

np.set_printoptions(formatter={'float_kind':"{:0.2f}".format})
# Choose which webcam to capture, 0 for default, 1 for external
#image = cv2.imread('C:\\Users\\profe\\OneDrive\\Skrivebord\\Github\\Group302-P3-Project\\dImages\\testImg.png')
def loadCamera():
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    return cap

def setCameraSize(cap):
    # Capture frame by frame
    ret, frame = cap.read()
    
    # Capture frame by frame
    return cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_AREA)
    
def load_database(verbose: bool = False) -> np.ndarray:
    """Load and preprocess the database of images"""
    database = np.zeros((N_IMAGES, int(IMG_WIDTH * IMG_HEIGHT)))  # Container for database
    if verbose:
        fig_list = []
        ax_list = []
        for _ in range(N_IMAGES):
            fig, ax = plt.subplots(1, 3)
            fig_list.append(fig)
            ax_list.append(ax)

    for i in range(N_IMAGES):
        img = cv2.imread(os.path.join(DATABASE_PATH, f'new_card{i}.jpg'))  # Read image in BGR (height, width, 3)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale (height, width)
        img_vector = img_gray.flatten()  # Convert image to vector (height * width)
        img_vector = img_vector / np.linalg.norm(img_vector)  # Normalize vector such that ||img_vector||_2 = 1
        database[i, :] = img_vector  

    return database

#while True:
    # Capture frame by frame
    #ret, frame = cap.read()

    # Resizing the webcam display size
    #frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_AREA)

def grayScale(frame):
    # Our operations on the frame come here
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Converting the current frame to gray

def threshholdImage(gray, tVal):
    return cv2.threshold(gray, tVal, 255, cv2.THRESH_BINARY_INV)

def otsuThreshholdImage(gray):
    return cv2.threshold(gray,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

def findContours(thresh_img):
    return cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

def compare(greyCrop,database):
    for i in range(N_IMAGES):
        img = cv2.imread(os.path.join(DATABASE_PATH, f'new_card{i}.png'))  # Read image in BGR (height, width, 3)
        img_vector = greyCrop.flatten()  # Convert image to vector (height * width)
        img_vector = img_vector / np.linalg.norm(img_vector)  # Normalize vector such that ||img_vector||_2 = 1
        dot_prod = np.dot(database, img_vector)  # Compute dot product b = A*x
        dot_prod = dot_prod / np.sum(dot_prod)  # Normalize dot product such that sum is 1
        # Print results:
        print(f'Input image had index: {i} OMP predicts index: {np.argmax(dot_prod)}, with "probability": {dot_prod[np.argmax(dot_prod)]}')

def drawContours(contours, frame, copiedFrame, database):
    for c in contours:
        cv2.drawContours(copiedFrame, [c], -1, (255,0,0), 3)
    
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
                cv2.putText(copiedFrame,(str(x)+','+str(y)),(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1,cv2.LINE_AA)
                cv2.imshow('Gaming frame', copiedFrame)
        
            # drawing skewed rectangle
            cv2.drawContours(copiedFrame, [approx], -1, (0, 255, 0))
            ##print("THIS IS CONTOURS",contours)
            if len(approx) == 4:
                print(approx)
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
                #ret, threshCrop = otsuThreshholdImage(greyCrop)
                cv2.imshow("Cropped grey", greyCrop)
                compare(greyCrop, database)
        except:
            print(" ")
    #elif c == 32: # Makes it so it only does the contour code when spacebar is clicked
        
        
if __name__ == '__main__':
    cap = loadCamera()
    database = load_database()
    while True:
        frame = setCameraSize(cap)
        copiedFrame = copy.copy(frame)
        gray = grayScale(frame)
        ret, thresh_img = threshholdImage(gray, 100)
        contours= findContours(thresh_img)
        #drawContours(contours, frame, copiedFrame)
        drawContours(contours, frame, copiedFrame, database)
        #contours = findContours(thresh_img)
        
        # Show the processed webcam feed
        cv2.imshow('Threshold frame', thresh_img)
        #cv2.imshow('Eroded', eroded)

        # Show the processed webcam feed
        cv2.imshow('Threshold frame', thresh_img)
        cv2.imshow('Camera frame', frame)
        #cv2.imshow('Transformed frame', dst)

        c = cv2.waitKey(1)
        if c == 27: #Press escape to exit
            break

    cap.release()
    cv2.destroyAllWindows()


