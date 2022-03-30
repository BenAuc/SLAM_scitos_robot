"""
@file hough_lines.py
@brief This program extracts line segments from the occupancy grid map using Hough transform and saves the set of lines
to a yaml file
Uncommenting the last section of the code will display the line segments overlaid on the image
"""
import sys
import cv2 as cv
import numpy as np
import yaml
import pickle
import matplotlib.pyplot as plt
    
filename = "/home/benoit/catkin_ws/src/ias0060_scitos_auclair_bryan_schneider/data/maps/map.pgm"

# Loads an image
src = cv.imread(filename)
# Check if image is loaded fine
if src is None:
    print('Error opening image!')
   
originalImage = src
grayImage = cv.cvtColor(originalImage, cv.COLOR_BGR2GRAY)

(thresh, blackAndWhiteImage) = cv.threshold(grayImage, 200, 255, cv.THRESH_BINARY)
invertedImage = ~blackAndWhiteImage

cv.imshow('Black white image', blackAndWhiteImage)
cv.imshow('Original image',originalImage)
cv.imshow('Gray image', grayImage)
cv.imshow('Iverted image', invertedImage)

edges = cv.Canny(grayImage, 50, 100, None, 3)
cv.imshow('Edges image', edges)

# cv.imwrite("data/maps/blackandwhite_map.png", blackAndWhiteImage)

# Copy edges to the images that will display the results in BGR
cdst = cv.cvtColor(grayImage, cv.COLOR_GRAY2BGR)
cdst0 = np.copy(cdst)
cdst1 = np.copy(cdst)

# linesP = cv.HoughLinesP(invertedImage, 2, np.pi / 180, 10, None, 5, 5)
linesFromBaW = cv.HoughLinesP(image=invertedImage, rho=3, theta=np.pi/2, threshold=3, lines=np.array([]), minLineLength=6, maxLineGap=2)
linesFromEdges = cv.HoughLinesP(image=edges, rho=1, theta=np.pi/2, threshold=3, lines=np.array([]), minLineLength=5, maxLineGap=1)

filename2 = "/home/benoit/catkin_ws/src/ias0060_scitos_auclair_bryan_schneider/data/config/map_features.yaml"

if linesFromBaW is not None:

    with open(filename2, 'w') as file:
        start_x = []
        start_y = []
        end_x = []
        end_y = []

        for idx in range(0, 1):
            l = linesFromBaW[idx][0]
            start_x.append(str(l[0]))
            start_y.append(str(l[1]))
            end_x.append(str(l[2]))
            end_y.append(str(l[3]))

            cv.line(cdst0, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 2, cv.LINE_AA)

        line_dict = {"start_x": start_x, "start_y": start_y, "end_x": end_x, "end_y": end_y}

        documents = yaml.dump(line_dict, file)

        
if linesFromEdges is not None:
    all_lines = dict
    for i in range(0, len(linesFromEdges)):
        print("coordinates line: ", i)

        l = linesFromEdges[i][0]
        print("coordinates start point: ", (l[0], l[1]))
        print("coordinates end point: ", (l[2], l[3]))
        cv.line(cdst1, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 2, cv.LINE_AA)


# cv.imshow("Source", src)
# cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
cv.imshow("Detected Lines - black and white img", cdst0)
cv.imshow("Detected Lines - edges", cdst1)

# plt.imshow(cdst0)
# plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
