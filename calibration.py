import numpy as np
import cv2
global edges, waitLength, ground_truth, faces_list
import os
import shutil

waitLength=0
cap = cv2.VideoCapture(1)
ground_truth = []
faces_list= []
boxes_list=[]
font = cv2.FONT_HERSHEY_SIMPLEX


def rect_trim(list_arg):
    # Recursively eliminates rectangles that overlap from the given list
    # Overlapping occurs if the distance between centers is less than width
    i = 0
    for rect in list_arg:
        current = rect
        if i > 0:
            prev = list_arg[i - 1]

            cx1, cy1 = current[0]
            cx2, cy2 = prev[0]

            d = cv2.norm((cx1, cy1), (cx2, cy2), cv2.NORM_L2)

            w, h = current[1]
            w2, h2 = prev[1]
            # trim for the external rectangle being detected as well.
            if d < w or w * h > 2 * (w2 * h2):
                list_arg.pop(i)
                i = i - 1
                rect_trim(list_arg)
        i = i + 1
    return list_arg, len(list_arg)

#input= contour
#output = list of minimum bounding rectangles around each sticker


def process_contours(contour_list):

    rectangles = []
    for cnt in contour_list:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 10:

            rect = cv2.minAreaRect(approx) #returns rotated rectangle occupying least area
            box = cv2.boxPoints(rect)
            box = np.int0(box) #gets 4 corners of rotated rectangle
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


    return rectangles


def process_img(input_frame):
    global picnumber, edges, waitLength
    img = cv2.flip(input_frame, 1)
    waitLength = 1
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    val = hsv[::, ::, 2]
    _, thresh = cv2.threshold(val, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3))

    erode = cv2.erode(thresh, kernel, iterations=2)
    edges = cv2.Canny(erode, 220, 230)

    _, contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect_list = process_contours(contours)

    rectangles, new_length = rect_trim(rect_list)

    ret, img, segment = process_rectangles(rectangles, img)

    return edges, img, ret, segment


def process_rectangles(rectlist, input_img):
    global ground_truth, waitLength, faces_list, picnumber
    new_length = len(rectlist)
    z=0
    segment = None
    flag = False
    if new_length == 9:
        waitLength = 0
        faces_list.append(input_img)

        flag = True
        rectlist.reverse()
        ijk = input_img.copy()
        for rect in rectlist:
            z+=1
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            center = tuple(np.int0(rect[0])) #retrieveing center of box
            cv2.drawContours(input_img, [box], 0, (0, 0, 255), 2)
            if z == 5:
                l, b = center
                segment= ijk[b-5:b+5, l-5:l+5]
                cv2.circle(input_img,center, 10, (0,255,0), -1)

            message2 = "Space: Skip frame, C: Calibrate, Q: Quit, R: Restart "

            cv2.putText(input_img, str(z), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(input_img, message2, (10, 470), font, 0.7, (255, 255, 255), 2)


    return flag, input_img, segment

#for i in range(1,7):

    #imname = 'photos/scramble'+str(i).zfill(3)+'.png'
    #img = cv2.imread(imname)

"""
Making a clean slate
"""
if os.path.isdir('photos'):
    shutil.rmtree('photos')
os.mkdir('photos')

picnumber = 0
while True:
    ret, img = cap.read()
    fname = 'photos/scramble' + str(picnumber).zfill(3) + '.png'
    #img= cv2.flip(img, 1)

    edges, output, retval, region = process_img(img)
    info = str(picnumber)+' faces calibrated.'
    if picnumber==6:
        info = info+' Press Q to Quit.'
    cv2.putText(output, info, (10, 450), font, 0.7, (255, 255, 255), 2)
    cv2.imshow('Identified Rectangles', output)
    cv2.imshow('Canny', edges)
    cv2.moveWindow('Canny', 700, 0)

    key= cv2.waitKey(waitLength)

    if key & 0xFF == ord('c') and retval and picnumber<6:
        ground_truth.append(region)
        if os.path.exists(fname):
            os.remove(fname)
        cv2.imwrite(fname, img)
        picnumber += 1
    elif key & 0xFF == ord('r'):
        boxes_list=[]
        ground_truth=[]
    elif key & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break



"""
Press C to calibrate.
Save image on calibration.
"""

cv2.destroyAllWindows()
