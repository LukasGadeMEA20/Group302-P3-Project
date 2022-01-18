# Libraries used
import cv2
import numpy as np
import copy
import math
import os
import imutils
import glob
import requests
from skimage import io   

#Width and height of images
IMG_WIDTH = 280
IMG_HEIGHT = 190

np.set_printoptions(formatter={'float_kind':"{:0.2f}".format})

# Function for getting the camera
def loadCamera():
    # Choose which webcam to capture, 0 for default, 1 and above for external
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    return cap

# Loads the database
def load_database(verbose: bool = False) -> np.ndarray:
    """Load and preprocess the database of images"""
    global database, onlyfiles # Global variables
    # Gets the jpg files
    images = [cv2.imread(file) for file in glob.glob("card_data_base/*.jpg")]
    # Gets the names of the files
    onlyfiles = [f for f in os.listdir("card_data_base/") if os.path.isfile(os.path.join("card_data_base/", f))]
    # Database for dot vectors of each image.
    database = np.zeros((len(onlyfiles), int(IMG_WIDTH * IMG_HEIGHT)))  # Container for database

    # For loop that goes through each image and adds it to the database
    for i in range(len(images)):
        img = images[i]
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to greyscale (height, width)
        img_vector = img_grey.flatten()  # Convert image to vector (height * width)
        img_vector = img_vector / np.linalg.norm(img_vector)  # Normalize vector such that ||img_vector||_2 = 1
        database[i, :] = img_vector  # Adds it to the database
        
# Gets the specified card from the database
def getMagicCard(card):
    # Makes a request from the API with the specific name by using "!"
    scryfallAPI = requests.get("https://api.scryfall.com/cards/search?q=!{}".format(card))
    # Makes sure the API is up and running
    if scryfallAPI.status_code == 200:
        # Saves it in JSON format
        scryfallJSON = scryfallAPI.json()
        # Gets the cards image
        url = scryfallJSON['data'][0]['image_uris']['border_crop']

        # Reads the url of the border crop
        cardToDisplay = io.imread(url)
        #Adjusts the colours
        cardToDisplay = cv2.cvtColor(cardToDisplay, cv2.COLOR_RGB2BGR)
        #Displays it
        cv2.imshow("Card",cardToDisplay)
    else:
        #Print error message if the API is not up and running
        print("api.scryfall:\n\tstatus_code:", scryfallAPI.status_code)

# Function that compares the image with the database
def compare(greyCrop):
    dot_prod = 0 # Dot product that it finds
    for i in range(len(onlyfiles)):
        img_vector = greyCrop.flatten()  # Convert image to vector (height * width)
        img_vector = img_vector / np.linalg.norm(img_vector)  # Normalize vector such that ||img_vector||_2 = 1
        dot_prod = np.dot(database, img_vector)  # Compute dot product b = A*x
        dot_prod = dot_prod / np.sum(dot_prod)  # Normalize dot product such that sum is 1
        # Print results:
        print(f'Input image had index: {i} OMP predicts index: {np.argmax(dot_prod)}, with "probability": {dot_prod[np.argmax(dot_prod)]}')
    return np.argmax(dot_prod) # Returns the dot product


# Variables for getting the click of the user
refPt = []
clicked = False

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
    global refPt
	# if the left mouse button was clicked, records the x and y coords
    # Then it checks if it within a card
    if event == cv2.EVENT_LBUTTONUP:
        refPt = [x,y]
        checkMousePoint(refPt)

# Adds it to the camera frame
cv2.namedWindow("Camera frame")
cv2.setMouseCallback("Camera frame", click_and_crop)

# Checks if it clicks on an object in the preprocessed image
def checkMousePoint(point):
    global clicked # Global variable
    # Checks if the value of the clicked position is 255 (white & an object)
    if im_out[point[1],point[0]] == 255:
        clicked = True
    else:
        clicked = False

# Sets the size of the camera
def setCameraSize(cap):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Method for greyscaling the image
def greyScale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Converting the current frame to grey

# Preprocessing part of the algorithm
def preProcess(greyScale, val):
    ret, im_th = threshholdImage(greyScale, val, cv2.THRESH_BINARY_INV) # Binarises the image based on a value

    floodBorder = copy.copy(im_th) # makes a copy of the binarised image to keep them seperate

    # Cleans up the border
    floodFill(floodBorder, 0,0,0)
    floodFill(floodBorder,0,0,255)

    # Inverses it to get the details
    im_floodfill_inv = cv2.bitwise_not(floodBorder)
    
    # Makes a global variable of the output and compares the two images
    # to get only the important details of the image
    global im_out
    im_out = im_floodfill_inv | im_th

    # Removes the final parts of the corners
    h, w = im_out.shape[:2]
    floodFill(im_out, 0,0, 0)
    floodFill(im_out, h-1, 0,0)
    floodFill(im_out, h-1, w-1,0)
    floodFill(im_out, 0, w-1,0)

    # Returns the corners
    return im_out

# Method for flooding out the pixels and changing the value
def floodFill(img, y, x, colour):
    h, w = img.shape[:2] # Gets the height and width of the image
    mask = np.zeros((h+2,w+2),np.uint8) # generates a mask based on the image
    # Fills out the pixels with the same colour
    # as the pixel at position x,y
    return cv2.floodFill(img, mask, (x,y), colour)  

# Function for quickly eroding with a kernel with size kSize x kSize
def erode(input, kSize):
    return cv2.erode(input, np.ones((kSize,kSize), np.uint8))

# Function for quickly Dilating with a kernel with size kSize x kSize
def dilate(input, kSize):
    return cv2.dilate(input, np.ones((kSize,kSize), np.uint8))

# Binarises the image based on a value and the type of binarisation method. 
def threshholdImage(grey, tVal, tType):
    return cv2.threshold(grey, tVal, 255, tType)

# Function for finding the contours on the image, based on the mode and method chosen
def findContours(thresh_img, mode, method):
    return cv2.findContours(thresh_img, mode, method)[-2]

# Function for selecting one of the contours
def selectOne(contours):
    # Variables for the center of the chosen contour
    global centerX, centerY, closeContours

    # Variable for checking each contour and checking which is closest to the mouse
    firstCheck = False
    closestDist = 0

    #Runs through each contour and gets their x and y to find which is closest to the mouse
    for c in contours:
        # x and y
        x = []
        y = []

        #Approximate contour as a rectangle
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * perimeter, True)
        approx = np.squeeze(approx) #Removes redundant dimension

        # For loop for each point in the rectangle
        for p in approx:
            x.append(p[0])
            y.append(p[1])

        # Finds the center point of x and y
        centerX = findMiddlePoint(x)
        centerY = findMiddlePoint(y)

        # Runs the distance check to check the distance
        dist = checkDistance(refPt, [centerX, centerY])
        # Checks if it closer than the previous closets. Runs if there is no previous closest.
        if closestDist > dist or firstCheck == False:
            closestDist = dist
            closeContours = c
            firstCheck = True
    
    # Returns the chosen contours
    return closeContours
         
# Finds the middle point by taking the average of the given array
def findMiddlePoint(array):
    temp = 0
    for i in array:
        temp+=i
    return temp/len(array)

# Checks the euler distance between the two given coords
def checkDistance(coord1, coord2):
    return math.dist(coord1, coord2)

# Method for rotating the image
def checkRotate(img):
    # copies the image
    imgToRotate = copy.copy(img)
    
    # Runs it four times as there are four sides and we do not wish for it to run infinitely
    # Could be upped to 8 or 16, to have it run extra times, but was deemed unecessary.
    for i in range(4):
        # Greyscales the image
        warpedGrey = greyScale(imgToRotate)

        # Gets the upper right corner by set variables
        manaSymbol = warpedGrey[5:35, 265:295]

        # binarises the corner and runs the blob analysis on it
        ret, manaSymbol_th = threshholdImage(manaSymbol, 100, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        manaBlob = blobFinder(manaSymbol_th, i, 100, 200, True, 300, False, 0.2, False, 0.6, False, 0)


        # Gets the bottom right corner by set variables
        manaSymbolReassurance = warpedGrey[360:395, 270:305]

        # binarises the corner and runs the blob analysis on it
        ret, manaSymbolReassurance_th = threshholdImage(manaSymbolReassurance, 100, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        manaBlobReassurance = blobFinder(manaSymbolReassurance_th, i+4, 100, 200, True, 400, False, 0.2, False, 0.6, False, 0)

        # Checks if only the upper right corner has blobs and returns the image
        if manaBlob != [] and manaBlobReassurance == []:
            return imgToRotate
        else: # else it will resize and rotate the image
            imgToRotate = cv2.resize(imgToRotate, [400,300])
            imgToRotate = imutils.rotate_bound(imgToRotate, 90)

# Function for finding blobs with each of the different parameters as attributes
def blobFinder(img, i, min_thresh, max_thresh, fBA, fBA_min, fBCi, fBCi_min, fBCO, fBCO_min, fBI, fBI_min):
    # Resizes the blob
    blob = cv2.resize(img, [100,100])
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = min_thresh
    params.maxThreshold = max_thresh

    # Filter by Area.
    params.filterByArea = fBA
    params.minArea = fBA_min

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

# algorithm for finding the specified card.
def findTheCard(c, frame, copiedFrame):
    #Visualization of the contours, debugging
    cv2.drawContours(copiedFrame, c, -1, (255,0,0), 3)

    #Approximate contour as a rectangle
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.01 * perimeter, True)
    approx = np.squeeze(approx) #Removes redundant dimension

    # Tries drawing the points for visualization
    try:
        #drawing points for visualisation, debugging
        for point in approx:
            x = point[0]
            y = point[1]
            cv2.circle(copiedFrame, (x, y), 3, (0, 255, 0), -1)
            cv2.putText(copiedFrame,(str(x)+','+str(y)),(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1,cv2.LINE_AA)
    except:
        print("Could not find card")

    
    # drawing skewed rectangle for visualization, debugging
    cv2.drawContours(copiedFrame, [c], -1, (0, 255, 0))

    # Runs this only if there are 4 sides to the figure.
    if len(approx) == 4:
        # Try to warp the card
        try:
            # Temporary empty array at the size of the what we want to card to have
            pts2 = np.float32([[0,0],[0,400],[300,400],[300,0]])
            
            # New matrix to transform the frame
            M = cv2.getPerspectiveTransform(approx.astype(np.float32),pts2)
            
            # Warped transform of the card
            warped = cv2.warpPerspective(frame,M,(300,400))

            # Try to rotate and display the card
            try:
                # Runs the check rotate function on the card, to try and fix the rotation.
                correctImage = checkRotate(warped)
                
                # Displays the rotated and warped card
                cv2.imshow('Transformed frame', correctImage)

                #Crop the image to get the artwork and show it
                croppedImg = correctImage[30:220, 10:290]
                # Note to devs. If above gets changed, the database images must also get changed
                # Displays the cropped image.
                cv2.imshow("Cropped image", croppedImg)

                # Try to compare the image to the database
                try:
                    #Preprocess the cropped image and show it
                    greyCrop = greyScale(croppedImg)
                    #Runs the compare function
                    card = compare(greyCrop)
                    # Gets the card from the API to display it.
                    getMagicCard(onlyfiles[card].replace(".jpg",""))
                except:
                    print("Cannot compare image") # Error catcher
            except:
                print("Cannot rotate image") # Error catcher
        except:
            print("Cannot warp persepctive") # Error catcher
        
# Runs only if it is not used in another code.
# Not as important as we do not have this code as a class
# But still good pracgtice to use.
if __name__ == '__main__':
    # - Setup phase - #
    cap = loadCamera() # Load the camera
    #Resizing the camera feed
    setCameraSize(cap)
    #Initialize the database
    load_database()

    # - Algorithm phase - #
    while True:
        # - Input - #
        ret, frame = cap.read()
        copiedFrame = copy.copy(frame)

        # - Pre-process - #
        grey = greyScale(frame) # Greyscales
        thresh_img = preProcess(grey, 123) # Preprocess at a value of 123 - during testing was found most efficient
        eroded = erode(thresh_img, 11) # Erodes with a large kernel of 11x11
        
        # - Segmentation, representation and classification - #
        # Only when the user has clicked on a viable card
        if clicked:
            # Finds the contours
            contours = findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Chooses the contour based on the selected cawrd
            cardClickedContours = selectOne(contours)
            # Finds the card
            findTheCard(cardClickedContours, frame, copiedFrame)
            # Makes sure it does not constantly run this part of the code again and again
            clicked = False
        
        # Feed for the user
        cv2.imshow('Camera frame', frame)

        # Adds a listener to escape to exit the program.
        c = cv2.waitKey(1)
        if c == 27: #Press escape to exit
            break

    # Once out of the while loop, it will release the camera from use and close all windows in order for it to exit the program.
    cap.release()
    cv2.destroyAllWindows()


