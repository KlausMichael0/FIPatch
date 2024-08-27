import numpy as np
import cv2
import random


def adjust_brightness_lab(img):
    brightness_factor = random.uniform(0.5, 1.5)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_img)
    l_channel = l_channel.astype(np.float32)
    l_channel *= brightness_factor
    l_channel = np.clip(l_channel, 0, 255)
    l_channel = l_channel.astype(np.uint8)
    adjusted_lab_img = cv2.merge((l_channel, a_channel, b_channel))
    adjusted_img = cv2.cvtColor(adjusted_lab_img, cv2.COLOR_LAB2BGR)
    return adjusted_img

def generate_perspective_matrices(w, h, num):
    perspective_matrices = []
    for _ in range(num):
        tl_offset = [w * random.uniform(0, 0.1), h * random.uniform(0, 0.1)]
        tr_offset = [w * (1 - random.uniform(0, 0.1)), h * random.uniform(0, 0.1)]
        br_offset = [w * (1 - random.uniform(0, 0.1)), h * (1 - random.uniform(0, 0.1))]
        bl_offset = [w * random.uniform(0, 0.1), h * (1 - random.uniform(0, 0.1))]
        perspective_matrix = np.float32([tl_offset, tr_offset, br_offset, bl_offset])
        perspective_matrices.append(perspective_matrix)
    perspective_matrices.append(np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]))
    return perspective_matrices

def adjust_sign_size_based_on_distance(img, actual_distance=4):
    min_distance = 1
    max_distance = 25
    distance = random.uniform(min_distance, max_distance)
    # print(distance)
    scaling_factor = actual_distance / distance *2
    h, w = img.shape[:2]
    new_w = int(w * scaling_factor)
    new_h = int(h * scaling_factor)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros_like(img)
    start_x = max((canvas.shape[1] - new_w) // 2, 0)
    start_y = max((canvas.shape[0] - new_h) // 2, 0)
    end_x = min(start_x + new_w, canvas.shape[1])
    end_y = min(start_y + new_h, canvas.shape[0])
    resized_start_x = max(0, (new_w - canvas.shape[1]) // 2)
    resized_start_y = max(0, (new_h - canvas.shape[0]) // 2)
    resized_end_x = resized_start_x + (end_x - start_x)
    resized_end_y = resized_start_y + (end_y - start_y)
    canvas[start_y:end_y, start_x:end_x] = resized_img[resized_start_y:resized_end_y, resized_start_x:resized_end_x]
    return canvas

def rotate_image(img):
    angle = random.uniform(-10, 10)
    # print(angle)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated_img

def motion_blur(image):
    angle = random.uniform(0, 180)
    length = random.randint(5, 20)
    angle = angle * np.pi / 180.0
    kernel_size = length
    kernel = np.zeros((kernel_size, kernel_size))
    cos_val = np.cos(angle)
    sin_val = np.sin(angle)
    mid = kernel_size // 2
    for i in range(kernel_size):
        x = i - mid
        y = int(round(x * sin_val / cos_val))
        y += mid
        if 0 <= y < kernel_size:
            kernel[mid, y] = 1.0
    kernel /= kernel_size
    blurred_image = cv2.filter2D(image, -1, kernel)
    return blurred_image

def apply_transparency(background_img, mask_img):
    alpha = random.uniform(0, 0.1)
    white_mask = np.zeros_like(mask_img)
    white_mask[mask_img > 0] = 255
    if background_img.shape[:2] != white_mask.shape[:2]:
        white_mask = cv2.resize(white_mask, (background_img.shape[1], background_img.shape[0]))
    white_mask = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
    composite_img = cv2.addWeighted(background_img, 1, white_mask, alpha, 0)
    return composite_img


def image_transformation(image, num):
    h, w, _ = image.shape
    perspective_mat = generate_perspective_matrices(w, h, num)
    res_images = []

    for i in range(num):
        # random brightness
        adv_img = adjust_brightness_lab(image)

        # random perspective
        before = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        after = perspective_mat[i]
        if (before - after).sum() != 0:
            matrix = cv2.getPerspectiveTransform(before, after)
            adv_img = cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
            adv_img = adv_img[int(min(after[0:2, 1])): int(max(after[2:4, 1])),
                      int(min(after[0::3, 0])): int(max(after[1:3, 0]))]

        # random distance
        adv_img = adjust_sign_size_based_on_distance(adv_img)

        # random rotation
        adv_img = rotate_image(adv_img)

        # random motion blur
        # adv_img = motion_blur(adv_img)

        # transparency transformation
        # adv_img = apply_transparency(adv_img, mask)

        res_images.append(adv_img)
    for i in range(num):
        res_images[i] = cv2.resize(res_images[i], (32, 32))
        # cv2.imshow(f"Image {i}", res_images[i])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    return res_images


