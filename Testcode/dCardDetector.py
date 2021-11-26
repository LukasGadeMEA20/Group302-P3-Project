import cv2
import numpy as np
import dCards

## Camera settings
IM_WIDTH = 1280
IM_HEIGHT = 720 
FRAME_RATE = 30

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

    #Preprocessing the video input (from dCards.py)
    preProc = dCards.preprocess_image(frame)

    # Find and sort the contours of all cards in the image (query cards)
    cnts_sort, cnt_is_card = dCards.find_cards(preProc)

    # If there are no contours, do nothing
    if len(cnts_sort) != 0:

        # Initialize a new "cards" list to assign the card objects.
        # k indexes the newly made array of cards.
        cards = []
        k = 0

        # For each contour detected:
        for i in range(len(cnts_sort)):
            if (cnt_is_card[i] == 1):

                # Create a card object from the contour and append it to the list of cards.
                # preprocess_card function takes the card contour and contour and
                # determines the cards properties (corner points, etc). It generates a
                # flattened 200x300 image of the card, and isolates the card's
                # suit and rank from the image.
                cards.append(dCards.preprocess_card(cnts_sort[i], frame))

                # Find the best rank and suit match for the card.
                #cards[k].best_rank_match,cards[k].best_suit_match,cards[k].rank_diff,cards[k].suit_diff = dCards.match_card(cards[k],train_ranks,train_suits)

                # Draw center point and match result on the image.
                image = dCards.draw_results(frame, cards[k])
                k = k + 1
        
        # Draw card contours on image (have to do contours all at once or
        # they do not show up properly for some reason)
        if (len(cards) != 0):
            temp_cnts = []
            for i in range(len(cards)):
                temp_cnts.append(cards[i].contour)
            cv2.drawContours(image,temp_cnts, -1, (255,0,0), 2)
    
    
    #Draw contours
    #contours =  cv2.findContours(preProc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    #for c in contours:
    #    cv2.drawContours(frame, [c], -1, (255,0,0), 3)

    #Show video stream
    cv2.imshow('Input', frame)
    #cv2.imshow('flattens',)

    c = cv2.waitKey(1)
    if c == 27: #Press escape to exit
        break

cap.release()
cv2.destroyAllWindows()

