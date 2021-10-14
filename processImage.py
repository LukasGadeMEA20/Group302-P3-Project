import cv2

def preprocess_image(image):
    """Returns a grayed, blurred, and adaptively thresholded camera image."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Converting the current frame to gray
    blur = cv2.GaussianBlur(gray,(5,5),0) #Blurring the grey frame
    retval, thresh_img = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    return thresh_img