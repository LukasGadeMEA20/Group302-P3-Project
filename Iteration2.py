import cv2
import numpy as np
import matplotlib as plt
import copy
import math

np.set_printoptions(formatter={'float_kind':"{:0.2f}".format})
# Choose which webcam to capture, 0 for default, 1 for external
#image = cv2.imread('C:\\Users\\profe\\OneDrive\\Skrivebord\\Github\\Group302-P3-Project\\dImages\\testImg.png')
def loadCamera():
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    return cap

refPt = []
clicked = False

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
    global refPt, cropping
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
    if event == cv2.EVENT_LBUTTONUP:
        refPt = [x,y]
        #print(str(refPt[1])+","+str(refPt[0]))
        checkMousePoint(refPt)
        #print(refPt)
        #clicked = True

cv2.namedWindow("Camera frame")
cv2.setMouseCallback("Camera frame", click_and_crop)

def checkMousePoint(point):
    global clicked
    #print(im_out[point[1],point[0]])
    if im_out[point[1],point[0]] == 255:
        clicked = True
    else:
        clicked = False

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

def preProcess(grayScale, val):
    ret, im_th = threshholdImage(grayScale, val)

    floodBorder = copy.copy(im_th)

    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Fill everything that is the same colour (black) as top-left corner with black
    cv2.floodFill(floodBorder, mask, (0,0), 0)

    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Fill everything that is the same colour (black) as top-left corner with white
    cv2.floodFill(floodBorder, mask, (0,0), 255)

    im_floodfill_inv = cv2.bitwise_not(floodBorder)

    global im_out
    im_out = im_floodfill_inv | im_th

    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Fill everything that is the same colour (black) as top-left corner with white
    cv2.floodFill(im_out, mask, (0,0), 0)

    return im_out

def erode(input, kSize):
    return cv2.erode(input, np.ones((kSize,kSize), np.uint8))

def dilate(input, kSize):
    return cv2.dilate(input, np.ones((kSize,kSize), np.uint8))

def threshholdImage(gray, tVal):
    return cv2.threshold(gray, tVal, 255, cv2.THRESH_BINARY_INV)

def otsuThreshholdImage(gray):
    return cv2.threshold(gray,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

def findContours(thresh_img):
    return cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

def selectOne(contours):
    global centerX, centerY, closeContours
    firstCheck = False
    closestDist = 0
    for c in contours:
        x = []
        y = []
        #Approximate contour as a rectangle
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * perimeter, True)
        approx = np.squeeze(approx) #Removes redundant dimension

        for p in approx:
            x.append(p[0])
            y.append(p[1])

        centerX = findMiddlePoint(x)
        centerY = findMiddlePoint(y)

        dist = checkDistance(refPt, [centerX, centerY])
        if closestDist > dist or firstCheck == False:
            closestDist = dist
            closeContours = c
            firstCheck = True
    
    return closeContours

            
def findMiddlePoint(array):
    temp = 0
    for i in array:
        temp+=i
    return temp/len(array)

def checkDistance(mousePosition, cardCenter):
    return math.dist(mousePosition, cardCenter)

def findFloodContours(thresh_img):
    return cv2.findContours(thresh_img, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)[-2]

def rotateImage(img):
    print("HI")
    print(img.shape)
    rows = img.shape[0]
    cols = img.shape[1]
    print("gaming")

    M = cv2.getRotationMatrix2D((cols/2, rows/2),-90,1)
    print("gaming1")
    return cv2.warpAffine(img,M,(cols,rows))

def rotateContours(c):
    p1, p2, p3, p4 = [], [], [], []
    tempContours = c
    newContours = c
    
    p1 = tempContours[0]
    for i in tempContours:
        if i[0] < p1[0]:
            p1 = i

    #print(tempContours, p1)
    indexArr = np.argwhere(tempContours == p1[1])
    tempContours = np.delete(tempContours, [indexArr], 0)

    p2 = tempContours[0]
    for i in tempContours:
        if distance(p1, i) < distance(p1, p2):
            p2 = i
    
    print(tempContours)
    indexArr = np.argwhere(tempContours == p2[0])
    print(indexArr)
    tempContours = np.delete(tempContours, [indexArr], 0)
    print(tempContours)

    print(p1[1], p2[1])
    newContours[1] = p1 if p1[1] < p2[1] else p2
    newContours[2] = p2 if p2[1] > p1[1] else p1

    print(tempContours)
    p3 = tempContours[0]
    p4 = tempContours[1]
    newContours[3] = p3 if p3[1] > p4[1] else p4
    newContours[0] = p4 if p3[1] > p4[1] else p3

    return newContours



def distance(p1, p2):
    return math.sqrt(math.pow((p2[1]-p1[1]),2) + math.pow((p2[0]-p1[0]),2))

def drawContours(c, frame, copiedFrame):
    #for c in contours:
    cv2.drawContours(copiedFrame, c, -1, (255,0,0), 3)

    #Approximate contour as a rectangle
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * perimeter, True)
    approx = np.squeeze(approx) #Removes redundant dimension

    #approx = rotateContours(approx)
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
            #print(approx)
            pts2 = np.float32([[0,0],[0,400],[300,400],[300,0]])
            M = cv2.getPerspectiveTransform(approx.astype(np.float32),pts2)
            #print(M)
            dst = cv2.warpPerspective(frame,M,(300,400))
            #dst = rotateImage(dst) this function rotates the image. 
            cv2.imshow('Transformed frame', dst)

            #Crop the image to get the artwork and show it
            croppedImg = dst[50:220, 30:270]
            cv2.imshow("Cropped image", croppedImg)

            #Preprocess the cropped image and show it
            greyCrop = grayScale(croppedImg)
            ret, threshCrop = otsuThreshholdImage(greyCrop)
            cv2.imshow("Cropped grey", threshCrop)
    except:
        print(" ")
    #elif c == 32: # Makes it so it only does the contour code when spacebar is clicked
        
        
if __name__ == '__main__':
    cap = loadCamera()
    while True:
        frame = setCameraSize(cap)
        copiedFrame = copy.copy(frame)
        gray = grayScale(frame)
        #ret, thresh_img = threshholdImage(gray, 100)
        thresh_img = preProcess(gray, 123)
        eroded = erode(thresh_img,11)
        #eroded = dilate(eroded,3)
        if clicked:
            contours = findContours(eroded)
            cardClickedContours = selectOne(contours)
            drawContours(cardClickedContours, frame, copiedFrame)
            clicked = False
        #contours = findContours(thresh_img)
        
        # Show the processed webcam feed
        cv2.imshow('Threshold frame', thresh_img)
        cv2.imshow('Eroded', eroded)

        # Show the processed webcam feed
        cv2.imshow('Threshold frame', thresh_img)
        cv2.imshow('Camera frame', frame)
        #cv2.imshow('Transformed frame', dst)

        c = cv2.waitKey(1)
        if c == 27: #Press escape to exit
            break

    cap.release()
    cv2.destroyAllWindows()


