import cv2
import numpy as np
import kociemba
import unicodedata
import clusterviz
import pickle
from MagicCube.code import cube_interactive
import matplotlib.pyplot as plt
import glob
global waitLength, faces

faces=[[], [], [], [], [], []]
waitLength = 1

answers = dict(clusterviz.final_labels)

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
    global edges, waitLength
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

    img = process_rectangles(rectangles, img)

    return edges, img


def process_rectangles(rectlist, input_img):
    global waitLength, faces
    new_length = len(rectlist)
    z=0
    stickers = []
    if new_length == 9:
        waitLength = 0
        rectlist.reverse()
        ijk = input_img.copy()
        ijk_lab = cv2.cvtColor(ijk, cv2.COLOR_BGR2LAB)
        for rect in rectlist:
            z+=1
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            (cx,cy) = tuple(np.int0(rect[0])) #retrieveing center of box
            midpixel= ijk_lab[cy, cx, :]
            midpixel= midpixel.reshape(1,-1)
            midpixel= midpixel.astype('float64')
            idx= clusterviz.model.predict(midpixel)[0]
            colour = answers[idx]
            stickers.append(idx)
            cv2.drawContours(input_img, [box], 0, (255, 255, 255), 2)
            cv2.putText(input_img, colour, (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        row = stickers[4]  # 5th index which is the center cube

        #reordering so that data fits Kociemba solver format
        for i in range(9):
            if i%3 == 0:
                stickers[i], stickers[i+2] = stickers[i+2], stickers[i]
        faces[row] = stickers


    return input_img


def refine(move_list):
    formatted_moves = []
    for m in move_list:
        if "'" in m:
            temp = (m[0], -1)
            formatted_moves.append(temp)
        elif '2' in m:
            temp = (m[0], 0)
            formatted_moves.append(temp)
            temp = (m[0], 0)
            formatted_moves.append(temp)
        else:
            temp = (m[0], 0)
            formatted_moves.append(temp)

    return formatted_moves


def get_stickers(list_arg):
    temp_list = [[],[],[],[],[],[]]
    """
    Reorder list so that
    Red = zeroth
    Green = first
    White = second
    Orange = third
    Blue = fourth
    Yellow = fifth
    """
    for each_face in list_arg:
        print each_face
        if each_face[4] == 'Red':
            temp_list[0] = each_face
        elif each_face[4] == 'Green':
            temp_list[1] = each_face
        elif each_face[4] == 'White':
            temp_list[2] = each_face
        elif each_face[4] == 'Orange':
            temp_list[3] = each_face
        elif each_face[4] == 'Blue':
            temp_list[4] = each_face
        elif each_face[4] == 'Yellow':
            temp_list[5] = each_face
    final_stickers = ''
    for each_face in temp_list:
        each_face = [e.replace('White', 'F') for e in each_face]
        each_face = [e.replace('Yellow', 'B') for e in each_face]
        each_face = [e.replace('Green', 'R') for e in each_face]
        each_face = [e.replace('Blue', 'L') for e in each_face]
        each_face = [e.replace('Red', 'U') for e in each_face]
        each_face = [e.replace('Orange', 'D') for e in each_face]
        stickers=''.join(each_face)
        final_stickers += stickers
    return final_stickers

for i in range(0,6):
    imname = 'photos/scramble'+str(i).zfill(3)+'.png'
    img = cv2.imread(imname)
    #img = cv2.flip(img, 1)
    edges, output = process_img(img)
    img= cv2.flip(img, 1)
    a,b,c= img.shape
    cv2.imshow('Output', output)
    cv2.imshow('Edges', edges)
    cv2.moveWindow('Edges', 700, 0)
    cv2.waitKey(waitLength)
    fname = 'detection'+str(i).zfill(3)+'.png'
    cv2.imwrite(fname, output)

cv2.destroyAllWindows()
final_results = [[],[],[],[],[],[]]
k=0
for each in faces:
    for e in each:
        final_results[k].append(answers[e])
    k+=1

current_state = get_stickers(final_results) #converts human readable format to Kociemba solver format
solved_state = 'UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB'
print current_state
original_scramble = kociemba.solve(solved_state, current_state)
original_scramble= unicodedata.normalize('NFKD', original_scramble).encode('ascii','ignore')
solution = kociemba.solve(current_state)
solution= unicodedata.normalize('NFKD', solution).encode('ascii','ignore')

default_face_colors = ["#cf0000", "#ff6f00",
                       "#00008f", "#009f0f",
                       "#ffcf00", "w",
                       "gray", "none"]

model= cube_interactive.Cube(N=3,face_colors=default_face_colors)

unrefined_moves = original_scramble.lower().split()

final_solution = refine(solution.lower().split())
final_scramble_moves= refine(unrefined_moves)



with open('MagicCubeSolution.p', 'wb') as f:
    pickle.dump(final_solution, f)

with open('Singmaster.p', 'wb') as f:
    pickle.dump(solution, f)

with open('Scramble.p', 'wb') as f:
    pickle.dump(final_scramble_moves, f)

for m in final_scramble_moves:
    if m[1] == 0:
        model.rotate_face(m[0].upper())
    elif m[1] == -1:
        model.rotate_face(m[0].upper(),-1)


print 'Original Scramble:', original_scramble

model.draw_interactive()
plt.show()

