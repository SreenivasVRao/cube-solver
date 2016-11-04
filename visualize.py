import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import calibration
from sklearn.externals import joblib
model = joblib.load('Kmeans2.pkl')
ret= True
name = 0

cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, img= cap.read()

    xerox = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    val = hsv[::, ::, 2]
    blur = cv2.GaussianBlur(val, (25, 25), 1)
    _, thresh = cv2.threshold(val, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    rect = np.ones((7, 7), np.uint8)

    kernel = np.ones((3, 3))

    erode = cv2.erode(thresh, kernel, iterations=2)
    edges = cv2.Canny(erode, 220, 230)
    _, contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    y = len(contours)
    squares = np.zeros(y)
    template = np.zeros((480, 640), dtype='uint8')
    rectangles = list()
    areas = []
    cnt_list = []
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 10:

            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            a = cv2.norm(box[0], box[1], cv2.NORM_L2)
            b = cv2.norm(box[1], box[2], cv2.NORM_L2)
            c = cv2.norm(box[2], box[3], cv2.NORM_L2)
            d = cv2.norm(box[3], box[0], cv2.NORM_L2)
            if b == 0 or d == 0:
                continue
            r1 = float(a) / float(b)
            r2 = float(c) / float(d)
            r3 = float(a * c) / float(b * d)
            if 0.9 <= r1 <= 1.1 and 0.9 <= r2 <= 1.1:
                rectangles.append(rect)

    boxes_list = []
    z = 0
    waitLength = 1
    rectangles, new_length = calibration.rect_trim(rectangles)
    if new_length == 9:
        waitLength = 0
        areas.append(cv2.contourArea(approx))

        rectangles.reverse()
        for rect in rectangles:
            z = z + 1
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            center = tuple(np.int0(rect[0]))
            font = cv2.FONT_HERSHEY_SIMPLEX
            x, y, w, h = cv2.boundingRect(box)
            ijk = img.copy()
            ijk_lab = cv2.cvtColor(ijk, cv2.COLOR_BGR2LAB)
            boxes_list.append(ijk_lab[y:y + h, x:x + w])
            a,b,c= ijk_lab.shape
            midpixel_value = ijk_lab[a / 2, b / 2, :]
            midpixel_value = midpixel_value.reshape([1, 3])
            midpixel_value.astype(float)

            colour = model.predict(midpixel_value)[0]
            # font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(colour), center, font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.drawContours(img, [box], 0, (255, 255, 255), 2)



    cv2.imshow("Display", img)
    cv2.imshow("Edges", edges)
    key = cv2.waitKey(waitLength)
    if  key & 0xFF == ord('q'):
        break



