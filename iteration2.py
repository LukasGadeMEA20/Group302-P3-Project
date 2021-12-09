import cv2
import numpy as np
import matplotlib as plt
import copy
import math
import os
import imutils
import inspect
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

refPt = []
clicked = False

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

def compare(greyCrop,database):
    for i in range(N_IMAGES):
        #img = cv2.imread(os.path.join(DATABASE_PATH, f'new_card{i}.png'))  # Read image in BGR (height, width, 3)
        img_vector = greyCrop.flatten()  # Convert image to vector (height * width)
        img_vector = img_vector / np.linalg.norm(img_vector)  # Normalize vector such that ||img_vector||_2 = 1
        dot_prod = np.dot(database, img_vector)  # Compute dot product b = A*x
        dot_prod = dot_prod / np.sum(dot_prod)  # Normalize dot product such that sum is 1
        # Print results:
        print(f'Input image had index: {i} OMP predicts index: {np.argmax(dot_prod)}, with "probability": {dot_prod[np.argmax(dot_prod)]}')

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
    global refPt
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
    if event == cv2.EVENT_LBUTTONUP:
        refPt = [x,y]
        checkMousePoint(refPt)

cv2.namedWindow("Camera frame")
cv2.setMouseCallback("Camera frame", click_and_crop)

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        print(str(x),str(y))

cv2.namedWindow("Transformed frame")
cv2.setMouseCallback("Transformed frame", click)

def checkMousePoint(point):
    global clicked
    #print(im_out[point[1],point[0]])
    if im_out[point[1],point[0]] == 255:
        clicked = True
    else:
        clicked = False

def setCameraSize(cap):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def grayScale(frame):
    # Our operations on the frame come here
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Converting the current frame to gray

def preProcess(grayScale, val):
    ret, im_th = threshholdImage(grayScale, val)

    floodBorder = copy.copy(im_th)    

    floodFill(floodBorder, 0,0,0)
    floodFill(floodBorder,0,0,255)

    # Kode der skal testes
    '''h, w = floodBorder.shape[:2]
    mask = np.zeros((h+2,w+2),np.uint8)
    
    for x in range(h):
        if floodBorder[x,0] != 0:
            cv2.floodFill(floodBorder, None, (0, x),0)
        if floodBorder[x, w-1] != 0:
            cv2.floodFill(floodBorder, None, (w-1, x),0)

    for y in range(w):
        if floodBorder[0,y] != 0:
            cv2.floodFill(floodBorder, None, (y,0),0)
        if floodBorder[h-1, y] != 0:
            cv2.floodFill(floodBorder, None, (y, h-1), 0)'''

    im_floodfill_inv = cv2.bitwise_not(floodBorder)

    global im_out
    im_out = im_floodfill_inv | im_th

    h, w = im_out.shape[:2]
    floodFill(im_out, 0,0, 0)
    floodFill(im_out, h-1, 0,0)
    floodFill(im_out, h-1, w-1,0)
    floodFill(im_out, 0, w-1,0)

    return im_out

def floodFill(img, y, x, colour):
    h, w = img.shape[:2]
    mask = np.zeros((h+2,w+2),np.uint8)
    return cv2.floodFill(img, mask, (x,y), colour)

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

#def rotateImage(img):
    #rows = img.shape[0]
    #cols = img.shape[1]

    #rotated = imutils.rotate_bound(img, 90)
    
    # grab the dimensions of the image and then determine the
    # center
    #(h, w) = img.shape[:2]
    #(cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    #M = cv2.getRotationMatrix2D((cX, cY), -90, 1.0)
    #cos = np.abs(M[0, 0])
    #sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    #nW = int((h * sin) + (w * cos))
    #nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    #M[0, 2] += (nW / 2) - cX
    #M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    #rotated = cv2.warpAffine(img, M, (nW, nH))

    #return rotated


def checkRotate(img):
    imgToRotate = copy.copy(img)
    for i in range(4):
        warpedGray = grayScale(imgToRotate)

        manaSymbol = warpedGray[5:35, 265:295]
        #cv2.imshow(("manaGray"+str(i)), manaSymbol)
        ret, manaSymbol_th = threshholdImage(manaSymbol, 70)
        manaSymbol_inv = cv2.bitwise_not(manaSymbol_th)
        manaBlob = blobFinder(manaSymbol_inv, i, 100, 200, True, 400, False, 0.2, False, 0.6, False, 0)

        manaSymbolReassurance = warpedGray[360:395, 270:305]
        ret, manaSymbolReassurance_th = threshholdImage(manaSymbolReassurance, 70)
        manaSymbolReassurance_inv = cv2.bitwise_not(manaSymbolReassurance_th)
        manaBlobReassurance = blobFinder(manaSymbolReassurance_inv, i+4, 100, 200, True, 500, False, 0.2, False, 0.6, False, 0)
        #setSymbol = warpedGray[230:250, 250:295]
        #setSymbolPreP = preProcess(setSymbol, 10)

        #symbolBlob = symbolBlobFinder(im_floodfill_inv, i)

        if manaBlob != [] and manaBlobReassurance == []:
            #cv2.imshow("LEZ GO"+str(i), imgToRotate)
            return imgToRotate
        else:
            imgToRotate = cv2.resize(imgToRotate, [400,300])
            imgToRotate = imutils.rotate_bound(imgToRotate, 90)

    #blob = cv2.resize(blob, [300,300])
    #cv2.imshow("BLOB",blob)

    #return imgToRotate

def blobFinder(img, i, min_thresh, max_thresh, fBA, fBA_min, fBCi, fBCi_min, fBCO, fBCO_min, fBI, fBI_min):
    blob = cv2.resize(img, [100,100])
    #print(cv2.calcHist(blob,[0],None,[256],[0,256]))
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = min_thresh
    params.maxThreshold = max_thresh

    # Filter by Area.
    params.filterByArea = fBA
    params.minArea = fBA_min
    # This value is what i tweaked to filter which areas it outlines
    # The value has to be 1602 or above to only outline the biggest blob.

    # Filter by Circularity
    params.filterByCircularity = fBCi
    params.minCircularity = fBCi_min

    # Filter by Convexity
    params.filterByConvexity = fBCO
    params.minConvexity = fBCO_min

    # Filter by Inertia
    params.filterByInertia = fBI
    params.minInertiaRatio = fBI_min

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(blob)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob

    im_with_keypoints = cv2.drawKeypoints(blob, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow(("image"+str(i)), im_with_keypoints)
    return keypoints

def distance(p1, p2):
    return math.sqrt(math.pow((p2[1]-p1[1]),2) + math.pow((p2[0]-p1[0]),2))

def drawContours(c, frame, copiedFrame):
    #for c in contours:
    cv2.drawContours(copiedFrame, c, -1, (255,0,0), 3)

    #Approximate contour as a rectangle
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.01 * perimeter, True)
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
            #cv2.imshow('Gaming frame', copiedFrame)
    except:
        print("Could not find card")

    try:
        # drawing skewed rectangle
        cv2.drawContours(copiedFrame, [c], -1, (0, 255, 0))
        if len(approx) == 4:
            pts2 = np.float32([[0,0],[0,400],[300,400],[300,0]])
            M = cv2.getPerspectiveTransform(approx.astype(np.float32),pts2)
            warped = cv2.warpPerspective(frame,M,(300,400))

            correctImage = checkRotate(warped)
            
            #rotated = rotateImage(warped) #this function rotates the image. 
            # Warp it to be the correct dimensions
            cv2.imshow('Transformed frame', correctImage)

            #Crop the image to get the artwork and show it
            croppedImg = correctImage[50:220, 30:270]
            cv2.imshow("Cropped image", croppedImg)

            #Preprocess the cropped image and show it
            greyCrop = grayScale(croppedImg)
            ret, threshCrop = otsuThreshholdImage(greyCrop)
            cv2.imshow("Cropped grey", threshCrop)
            compare(greyCrop, database)
    except:
        print("Error 2")    
    
    
    #elif c == 32: # Makes it so it only does the contour code when spacebar is clicked
        
        
if __name__ == '__main__':
    # - Setup phase - #
    cap = loadCamera() # Load the camera
    #Resizing the camera feed
    setCameraSize(cap)
    #Initialize the database and set it as a global variable
    global database
    database = load_database()


    while True:
        #frame = setCameraSize(cap)
        ret, frame = cap.read()
        copiedFrame = copy.copy(frame)
        gray = grayScale(frame)
        #ret, thresh_img = threshholdImage(gray, 100)
        thresh_img = preProcess(gray, 123)
        eroded = erode(thresh_img,11)
        #dilated = dilate(eroded,3)
        if clicked:
            contours = findContours(eroded)
            cardClickedContours = selectOne(contours)
            drawContours(cardClickedContours, frame, copiedFrame)
            clicked = False
        
        cv2.imshow('Camera frame', frame)

        c = cv2.waitKey(1)
        if c == 27: #Press escape to exit
            break

    cap.release()
    cv2.destroyAllWindows()


