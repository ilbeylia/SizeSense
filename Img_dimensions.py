import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

"""
TODO 
Take contour information one by one and check the corners of those containing 4 corners, perform perspective adjustment accordingly, 
 then scale and obtain the dimensions of the object in the middle
"""

def img_dim(img_path):
    temp_img = cv.imread(img_path)
    ret, temp_img_Threshold = cv.threshold(temp_img, 126, 255, cv.THRESH_BINARY)

    LW_Color = np.array([0, 0, 0])
    UP_Color = np.array([50, 50, 50])
    temp_mask = cv.inRange(temp_img_Threshold, LW_Color, UP_Color)

    contours, _ = cv.findContours(temp_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    all_corners = np.int0(cv.goodFeaturesToTrack(temp_mask, 100, 0.01, 10))
    print(len(contours))

    selected_corners = corner_select(all_corners)

    return temp_img, selected_corners, contours


def corner_select(corners):
    x1, y1 = np.min(corners, axis=0)[0]
    x2, y2 = np.max(corners, axis=0)[0]

    selected_corner = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]

    return selected_corner


def contours_Area(Contours):
    Total_Area = 0
    for contour in Contours:
        Total_Area = Total_Area + cv.contourArea(contour)
    
    return Total_Area




#TODO ---create tkinter window   
img_path = str(input("Img file path:"))
temp_img_path = "/home/ilbeyli/Desktop/Image_Dimensions_Process/img_dim_temp.png"

    
temp_Img, temp_Corners, temp_Contours = img_dim(temp_img_path)

Dims_temp = cv.rectangle(temp_Img, temp_Corners[0],temp_Corners[3], (0, 255,0 ), 2 )

cv.drawContours(temp_Img, temp_Contours, -1, (0, 255, 0), 2)

temp_Total_Area = contours_Area(temp_Contours)  # template visual black areas 
temp_Side_Lenght = abs(temp_Corners[0][1] - temp_Corners[1][1])

if img_path != "":
    img, img_Corners, img_Contours = img_dim(img_path)
    cv.drawContours(img, img_Contours, -1, (0, 255, 0), 1)
    img = cv.rectangle(img, img_Corners[0], img_Corners[3], (0,0,255),2)

    print(img_Corners)
    plt.imshow(img)
    plt.title("img")
    plt.show()


print("Side Lenght:", temp_Side_Lenght)
print("Total Area:", temp_Total_Area)



if Dims_temp is not None:
    plt.imshow(Dims_temp)
    plt.title("Corner_Draw")
    plt.show()
