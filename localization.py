import cv2 as cv
import numpy as np
import random
from torchvision import transforms, datasets, models


# input_size = (299, 299)
input_size = (32, 32)
transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ])

def get_block_coordinates(x, y):
    block_size = 32 // 8
    block_x = int(x // block_size)
    block_y = int(y // block_size)
    return block_x, block_y

def count_blocks(image_coordinates):
    block_counts = np.zeros((8, 8), dtype=int)
    total_points = len(image_coordinates)
    for coord in image_coordinates:
        block_x, block_y = get_block_coordinates(coord[0], coord[1])
        block_counts[block_y][block_x] += 1
    block_ratios = block_counts / total_points
    block_counts *= 20
    return block_counts, block_ratios

def contrast_enhance(img):
    img_lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    L, a, b = cv.split(img_lab)
    L = cv.equalizeHist(L)
    img_lab_merge = cv.merge((L, a, b))
    return cv.cvtColor(img_lab_merge, cv.COLOR_Lab2BGR)

def auto_canny(img, method, sigma=0.33):
    if method == "median":
        Th = np.median(img)

    elif method == "triangle":
        Th, _ = cv.threshold(img, 0, 255, cv.THRESH_TRIANGLE)

    elif method == "otsu":
        Th, _ = cv.threshold(img, 0, 255, cv.THRESH_OTSU)

    else:
        raise Exception("method specified not available!")
    lowTh = (1 - sigma) * Th
    highTh = (1 + sigma) * Th
    return cv.Canny(img, lowTh, highTh, apertureSize=3, L2gradient=True), highTh

# Ranges of HSV color spaces (red, blue, yellow and black)
lower_red1 = (0, 40, 50)
upper_red1 = (10, 255, 210)
lower_red2 = (165, 40, 50)
upper_red2 = (179, 255, 210)
lower_blue = (90, 40, 50)
upper_blue = (120, 255, 210)
lower_yellow = (20, 40, 50)
upper_yellow = (35, 255, 210)
lower_black = (0, 0, 0)
upper_black = (179, 255, 5)

def color_seg(img, kernel_size=None):
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask_red1 = cv.inRange(hsv_img, lower_red1, upper_red1)
    mask_red2 = cv.inRange(hsv_img, lower_red2, upper_red2)
    mask_blue = cv.inRange(hsv_img, lower_blue, upper_blue)
    mask_yellow = cv.inRange(hsv_img, lower_yellow, upper_yellow)
    mask_black = cv.inRange(hsv_img, lower_black, upper_black)
    mask_combined = mask_red1 | mask_red2 | mask_blue | mask_yellow | mask_black
    if kernel_size is not None:
        kernel = np.ones(kernel_size, np.uint8)
    else:
        kernel = np.ones((3, 3), np.uint8)
    mask_combined = cv.morphologyEx(mask_combined, cv.MORPH_OPEN, kernel)
    mask_combined = cv.morphologyEx(mask_combined, cv.MORPH_CLOSE, kernel)
    return mask_combined

def cnt_rect(cnts, coef=0.1):
    contour_list = []
    for cnt in cnts:
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, coef * peri, True)
        if len(approx) == 4:
            contour_list.append(cnt)

    if not contour_list:
        return None
    else:
        LC = max(contour_list, key=cv.contourArea)
        return LC

def cnt_circle(img, hough_dict):
    mask = np.zeros_like(img)
    circles = cv.HoughCircles(img,
                              cv.HOUGH_GRADIENT,
                              hough_dict["dp"],
                              hough_dict["minDist"],
                              param1=hough_dict["param1"],
                              param2=hough_dict["param2"],
                              minRadius=hough_dict["minRadius"],
                              maxRadius=hough_dict["maxRadius"])
    if circles is None:
        return circles
    else:
        list_circles = circles[0]
        largest_circles = max(list_circles, key=lambda x: x[2])
        center_x, center_y, r = largest_circles
        cv.circle(mask, (int(center_x), int(center_y)), int(r), 255)
        cnts = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnt = cnts[0]
        if len(cnts[0]) > 0:
            return max(cnt, key=cv.contourArea)
        else:
            return cnt[-1]

def integrate_circle_rect(rect_cnt, circle_cnt, cnt):
    if circle_cnt is not None and rect_cnt is not None:
        if cv.contourArea(circle_cnt) >= cv.contourArea(rect_cnt):
            output = circle_cnt
        else:
            output = rect_cnt
    elif circle_cnt is not None and rect_cnt is None:
        output = circle_cnt
    elif circle_cnt is None and rect_cnt is not None:
        output = rect_cnt
    else:
        if len(cnt) == 0:
            return np.array([])
        else:
            output = max(cnt, key=cv.contourArea)
    return output

def integrate_edge_color(output1, output2):
    if not isinstance(output1, np.ndarray):
        output1 = np.array(output1)
    if not isinstance(output2, np.ndarray):
        output2 = np.array(output2)
    if len(output1) == 0 and len(output2) == 0:
        return np.array([])
    elif len(output1) == 0 and output2.shape[-1] == 2:
        return output2
    elif len(output2) == 0 and output1.shape[-1] == 2:
        return output1
    else:
        if cv.contourArea(output1[0]) > cv.contourArea(output2[0]):
            return output1
        else:
            return output2

def generate_random_red():
    red = random.randint(0, 255)
    green = 0
    blue = 0
    return red, green, blue

