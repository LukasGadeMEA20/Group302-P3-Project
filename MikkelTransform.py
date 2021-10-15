import cv2
import numpy as np
import matplotlib as plt 

# Choose which webcam to capture, 0 for default, 1 for external
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

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

    #meanThresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 115, 1) #Applys adaptive threshold to the grayscale image
    #gauss = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1) #Applys adaptive threshold to the grayscale image
    retval2,otsu = cv2.threshold(gray,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    contours =  cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    for c in contours:
        cv2.drawContours(frame, [c], -1, (255,0,0), 3)
    
        #Approximate contour as a rectangle
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * perimeter, True)
<<<<<<< Updated upstream
=======
        print("approx",approx)
        print("Perimeter",perimeter)
        approx = np.squeeze(approx) #Removes redundant dimension
>>>>>>> Stashed changes

       # drawing points
        for point in approx:
            x, y = point[0]
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            cv2.putText(frame,str(x),(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1,cv2.LINE_AA)

        # drawing skewed rectangle
        cv2.drawContours(frame, [approx], -1, (0, 255, 0))

<<<<<<< Updated upstream
    img = cv2.imread('serraangelcrop.jpg')
    img = frame
    rows,cols,ch = img.shape

    pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    pts2 = np.float32([[0,0],[200,0],[0,300],[x,y]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(img,M,(300,300))
    
=======

    pts2 = np.float32([[0,0],[0,300],[300,300],[300,0]])

    M = cv2.getPerspectiveTransform(approx.astype(np.float32),pts2)
    dst = cv2.warpPerspective(frame,M,(300,300))
        
>>>>>>> Stashed changes
    # Show the processed webcam feed
    cv2.imshow('Threshold frame', thresh_img)
    cv2.imshow('Camera frame', frame)
    cv2.imshow('Transformed frame', dst)

    c = cv2.waitKey(1)
    if c == 27: #Press escape to exit
        break

cap.release()
cv2.destroyAllWindows()