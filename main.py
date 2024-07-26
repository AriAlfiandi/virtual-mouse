from cvzone.HandTrackingModule import HandDetector
import cv2
import math
from pynput.mouse import Controller, Button
import numpy as np
import screeninfo

# Get screen resolution
screen = screeninfo.get_monitors()[0]
screen_width, screen_height = screen.width, screen.height

mouse = Controller()
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.3, maxHands=1)

while True:
    success, img = cap.read()
    if not success:
        break

    # Flip the image horizontally
    img = cv2.flip(img, 1)

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        if lmList:
            x1, y1 = lmList[4][0], lmList[4][1]    # Thumb tip
            x2, y2 = lmList[8][0], lmList[8][1]    # Index finger tip
            x3, y3 = lmList[12][0], lmList[12][1]  # Middle finger tip
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Draw circles and lines on the hand landmarks
            cv2.circle(img, (x1, y1), 5, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 5, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Calculate distances between thumb and index, and thumb and middle finger
            length = math.hypot(x2 - x1, y2 - y1)
            middle_thumb_length = math.hypot(x3 - x1, y3 - y1)

            # Scroll mouse based on thumb and index finger distance
            if length < 50:
                mouse.scroll(0, 1)
                cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
            elif length > 150:
                mouse.scroll(0, -1)
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
            else:
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # Click mouse if middle finger and thumb tips are close
            if middle_thumb_length < 40:
                mouse.click(Button.left, 1)
                cv2.circle(img, (x3, y3), 5, (0, 255, 255), cv2.FILLED)

            # Move mouse cursor with index finger
            screen_x = int(np.interp(x2, [0, img.shape[1]], [0, screen_width]))
            screen_y = int(np.interp(y2, [0, img.shape[0]], [0, screen_height]))
            mouse.position = (screen_x, screen_y)

    cv2.imshow("Frame", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
