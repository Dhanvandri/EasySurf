import numpy as np
import cv2 as cv
import math
import pyautogui

cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    cv.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 0)
    crop_img = frame[100:300, 100:300]
    blur = cv.GaussianBlur(crop_img, (3, 3), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    mask1 = cv.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))
    kernal = np.ones((5, 5))
    dilation = cv.dilate(mask1, kernal, iterations=1)
    erosion = cv.erode(dilation, kernal, iterations=1)
    filtered = cv.GaussianBlur(erosion, (3, 3), 0)
    ret, thresh = cv.threshold(filtered, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    try:
        contour = max(contours, key=lambda x: cv.contourArea(x))
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 0)
        hull = cv.convexHull(contour)
        drawing = np.zeros(crop_img.shape, np.uint8)
        cv.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
        cv.drawContours(drawing, [hull], -1, (0, 0, 255), 0)
        hull = cv.convexHull(contour, returnPoints=False)
        defects = cv.convexityDefects(contour, hull)
        count = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

            if angle <=90:  
                count += 1
                cv.circle(crop_img, far, 5, [0, 0, 255], -1)
            cv.line(crop_img, start, end, [0, 255, 0], 2)

        if count == 1:
            if start[1] > end[1] and start[1] > far[1]:
                cv.putText(frame, "Left", (100, 80), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                pyautogui.press('left')
        elif count == 2:
            if start[0] > end[0] and start[0] > far[0] and end[0] > far[0]:
                cv.putText(frame, "Right", (100, 80), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                pyautogui.press('right')

        elif count >= 3:
            cv.putText(frame, "Stop", (100, 80), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            pyautogui.press('up')

    except:
        pass

    cv.imshow('EasySurf', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
