import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

"""
TODO 
Take contour information one by one and check the corners of those containing 4 corners, perform perspective adjustment accordingly, 
 then scale and obtain the dimensions of the object in the middle
"""

def img_Dim_Config(img_path):
    temp_img = cv.imread(img_path)
    ret, temp_img_Threshold = cv.threshold(temp_img, 126, 255, cv.THRESH_BINARY)

    LW_Color = np.array([0, 0, 0])
    UP_Color = np.array([50, 50, 50])
    temp_mask = cv.inRange(temp_img_Threshold, LW_Color, UP_Color)

    contours, _ = cv.findContours(temp_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    all_corners = np.intp(cv.goodFeaturesToTrack(temp_mask, 100, 0.01, 10))



    return temp_img, all_corners, contours


def temp_Corrner_Select(corners):
    x1, y1 = np.min(corners, axis=0)[0]
    x2, y2 = np.max(corners, axis=0)[0]

    selected_corner = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]

    return selected_corner


def contours_Area(Contours):
    Total_Area = 0
    for contour in Contours:
        Total_Area = Total_Area + cv.contourArea(contour)
    
    return Total_Area

def img_Corner_Select(all_Contours):
    Select_Contours = []
    for a in range(len(all_Contours)):
        if len(all_Contours[a]) >= 6:
            Select_Contours.append(all_Contours[a])    

    x1, _, y1, _ = x_y_MaxAndMin(Select_Contours[0])
    x2, _, _, y2 = x_y_MaxAndMin(Select_Contours[1])
    _, x3, y3, _ = x_y_MaxAndMin(Select_Contours[2])
    _, x4, _, y4 = x_y_MaxAndMin(Select_Contours[3])
    
    # Select_Corners = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    Select_Corners = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    
    return  Select_Corners

def x_y_MaxAndMin(Contours):

    y_values = [ values[0][1] for values in Contours]

    y_min = min(y_values)
    y_max = max(y_values)

    x_min = min([ values[0][0] for values in Contours if values[0][1] == y_min])
    x_max = max([ values[0][0] for values in Contours if values[0][1] == y_max])

    return x_min, x_max, y_min, y_max


def perspective_correction(img, corners, reference_size):
    reference_corners = np.float32([[0, 0], [0, reference_size[1]], [reference_size[0], reference_size[1]], [reference_size[0], 0]])

    homography_matrix, _ = cv.findHomography(corners, reference_corners)

    corrected_image = cv.warpPerspective(img, homography_matrix, reference_size)

    return corrected_image

#TODO ---create tkinter window   
img_path = str(input("Img file path:"))
temp_img_path = "img_dim_temp.png"

    
temp_Img, temp_all_corner, temp_Contours = img_Dim_Config(temp_img_path)
temp_Corners = temp_Corrner_Select(temp_all_corner)
Dims_temp = cv.rectangle(temp_Img, temp_Corners[0],temp_Corners[3], (0, 255,0 ), 2 )

cv.drawContours(temp_Img, temp_Contours, -1, (0, 255, 0), 2)

temp_Total_Area = contours_Area(temp_Contours)  # template visual black areas 
temp_Side_Lenght = abs(temp_Corners[0][1] - temp_Corners[1][1])

reference_size = (temp_Side_Lenght, temp_Side_Lenght)  # Since it is square, the length and width are the same


if img_path != "":
    img, img_Corners, img_Contours = img_Dim_Config(img_path)
    cv.drawContours(img, img_Contours, -1, (0, 255, 0), 1)

    Find_Corners = img_Corner_Select(img_Contours)
    # img = cv.rectangle(img, img_Corners[0], img_Corners[3], (0,0,255),2)
    Corrected_Img = perspective_correction(img, Find_Corners, reference_size)

    print(Find_Corners)
    plt.imshow(Corrected_Img)
    plt.title("img")
    plt.show()


# print(temp_Corners)
# print(temp_Contours)
# print("Side Lenght:", temp_Side_Lenght)
# print("Total Area:", temp_Total_Area)



if Dims_temp is not None:
    plt.imshow(Dims_temp)
    plt.title("Corner_Draw")
    plt.show()
