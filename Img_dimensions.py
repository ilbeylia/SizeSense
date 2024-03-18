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

def img_Corner_Select(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)
    dst = cv.dilate(dst, None)

    threshold = 0.01 * dst.max()
    corner_coordinates = np.where(dst > threshold)

    corner_points = np.column_stack((corner_coordinates[1], corner_coordinates[0]))

    find_Corners = []
    tolerance = 60  # ??? 
    index = 0
    # corner filter 
    for x, y in corner_points:
        if len(find_Corners) == 0:
            find_Corners.append((x, y))
        else:
            if index < len(find_Corners) and tolerance < abs(find_Corners[index][0] - x):
                index += 1
                find_Corners.append((x, y))
            # if index < len(find_Corners) and tolerance < abs(find_Corners[index][1] - y):
            #     index += 1
            #     find_Corners.append((x, y))
        
    for x,y in find_Corners:
        cv.circle(img, (x, y), 5, (0, 0, 255), -1)
    return find_Corners


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

    Find_Corners = img_Corner_Select(img)
    # for index, corner in enumerate(Find_Corners):
    #     cv.circle(img, corner, 5, (0, 255, 0), -1)
    #     cv.putText(img,str(index), corner, cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv.LINE_AA )

    # img = cv.rectangle(img, Find_Corners[1], Find_Corners[0], (0,0,255),2)
    # Corrected_Img = perspective_correction(img, Find_Corners, reference_size)


    print(len(Find_Corners))
    plt.imshow(img)
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
