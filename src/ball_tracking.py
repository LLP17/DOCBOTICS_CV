import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert From The OpenCV BGR colour space to the HSV Colour space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create the upper and lower thresholds to describe the colour yellow of interest in the HSV colour space. 
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    # Create a mask to filter all the pixels outside of the colour range described
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Find the contours of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)

        ((x, y), radius) = cv2.minEnclosingCircle(largest)

        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0,255,0), 2)
            cv2.circle(frame, (int(x), int(y)), 5, (0,0,255), -1)

            print(f"Ball position: {int(x)}, {int(y)}")

    cv2.imshow("Ball Tracker", frame)
    cv2.imshow("Mask Frame", mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

