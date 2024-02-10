import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0) #assigning a variable to the video input

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Managing FPS
pTime = 0
cTime = 0

while True:
    success, img = cap.read() #assigning a variable to the read video imput, this si the main input var now
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB) #basically creating an object up until now with all the hand data
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks: #for each hand use the mpDeaw function to create landmarks and connections using the HAND_CONNECTIONS
            for handLms in results.multi_hand_landmarks:
                thumb_tip_y = handLms.landmark[4].y  # THUMB_TIP
                thumb_mcp_y = handLms.landmark[2].y # THUMB_MCP
                index_tip_y = handLms.landmark[8].y  # INDEX_FINGER_TIP
                middle_tip_y = handLms.landmark[12].y  # MIDDLE_FINGER_TIP
                ring_tip_y = handLms.landmark[16].y  # RING_FINGER_TIP
                pinky_tip_y = handLms.landmark[20].y  # PINKY_TIP

                if thumb_tip_y < index_tip_y and thumb_tip_y < middle_tip_y \
                and thumb_tip_y < ring_tip_y and thumb_tip_y < pinky_tip_y and thumb_tip_y < thumb_mcp_y:   
                    cv.putText(img, "THUMBS UP", (50, 90), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
                    cv.imshow('Image', img)
                
                elif thumb_tip_y > index_tip_y and thumb_tip_y > middle_tip_y \
                and thumb_tip_y > ring_tip_y and thumb_tip_y > pinky_tip_y and thumb_tip_y > thumb_mcp_y:
                    cv.putText(img, "THUMBS DOWN", (50, 90), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
                    cv.imshow('Image', img)  

                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        
    cTime = time.time()
    fps = 1 / ( cTime - pTime)         
    pTime = cTime       

    cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)

    cv.imshow('Image', img)
        
    if cv.waitKey(20) & 0xFF==ord('d'):
            break   

cap.release()
cv.destroyAllWindows()
    