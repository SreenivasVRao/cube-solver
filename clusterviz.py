import cv2
import numpy as np
import collections
from sklearn.cluster import KMeans

import calibration
training_data = calibration.ground_truth
test_data = calibration.faces_list

np.save('raw_image_data.npy', training_data)
pixel_values = None

colours = ['White', 'Yellow', 'Green', 'Blue', 'Red', 'Orange']

#data pre-processing step
for each in training_data:
    lab_img = cv2.cvtColor(each, cv2.COLOR_BGR2LAB, 3)

    temp= lab_img.transpose(2,0,1).reshape(3,-1)
    #http://stackoverflow.com/questions/32838802/numpy-with-python-convert-3d-array-to-2d
    if pixel_values is not None:
        pixel_values=np.concatenate((pixel_values, temp), axis = 1)
    else: pixel_values= temp
pixel_values= pixel_values.swapaxes(0,1)
np.save('lab_data_points.npy', pixel_values)
#K Means modelling
model = KMeans(n_clusters=6)
model.fit(pixel_values)
labels = model.labels_
final_labels=[]
k=0
for i in range(0, 600, 100):
    x = np.array(collections.Counter(labels[i:i+100]).items())
    a = x[:, 0]
    b = x[:, 1]
    final_labels.append(a[b.argmax()])

    k+=1

final_labels= zip(final_labels, colours)
