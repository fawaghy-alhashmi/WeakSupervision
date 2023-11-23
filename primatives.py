import cv2
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

def area(mask_path, dpi_value=72):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        total_area += area

    area_mm2 = total_area * (1 / dpi_value) ** 2

    return area_mm2

def diameter(area_mm2): #Classic way, but since the lesions are irregular try other ways like feret
    diameter = 2 * math.sqrt(area_mm2 / math.pi)
    return diameter

def feret_diameter(mask_path, dpi_value=72):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_feret_diameter = 0

    for contour in contours:
        if len(contour) > 1:
            ymin, xmin = np.min(contour.squeeze(), axis=0)
            ymax, xmax = np.max(contour.squeeze(), axis=0)
            diameter = np.linalg.norm([ymax - ymin, xmax - xmin])

            if diameter > max_feret_diameter:
                max_feret_diameter = diameter

    max_feret_diameter_mm = max_feret_diameter / dpi_value

    return max_feret_diameter_mm

def feret_diameter2(mask_path, dpi_value=72):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_feret_diameter = 0

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        diameter = max(w, h)

        if diameter > max_feret_diameter:
            max_feret_diameter = diameter

    max_feret_diameter_mm = max_feret_diameter / dpi_value

    return max_feret_diameter_mm

def variance(image_path, mask_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    masked_img = cv2.bitwise_and(img, img, mask=mask)
    variance = np.var(masked_img)

    return variance

def masked_image(image_path,mask_path):

    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    masked_img = cv2.bitwise_and(img, img, mask=mask)

    return masked_img


def lab_image(masked_img):
    lab_image = cv2.cvtColor(masked_img, cv2.COLOR_BGR2LAB)

    avg_color_per_row = np.average(lab_image, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    avg_l, avg_a, avg_b = avg_color

    return avg_l,avg_a, avg_b


def naive_asymmetry_convexity_defects(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, threshold = cv2.threshold(gray, 0, 255, (cv2.THRESH_BINARY + cv2.THRESH_OTSU))
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        
        hull = cv2.convexHull(max_contour, returnPoints=False)
        defects = cv2.convexityDefects(max_contour, hull)

        if defects is not None and len(defects) > 13:  #Threshold set to 13 by trial and error
            return "Asymmetric"
        else:
            return "Symmetric"
    else:
        return "Symmetric" #if no contours

def asymmetry_convexity_defects(image_path): #Update to the asymmetry_convexity_defects method
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]  
        max_contour = max(contours, key=cv2.contourArea)
        
        if len(max_contour) > 2:
            mask = np.zeros_like(threshold)
            cv2.drawContours(mask, [max_contour], -1, (255), thickness=cv2.FILLED)

            segmented_lesion = cv2.bitwise_and(threshold, mask)
            inverted_mask = cv2.bitwise_not(mask)
            filled_external_area = cv2.bitwise_xor(segmented_lesion, inverted_mask)

            # Removing border
            kernel = np.ones((5, 5), np.uint8)
            filled_external_area = cv2.erode(filled_external_area, kernel, iterations=1)

            hull = cv2.convexHull(max_contour, returnPoints=False)
            defects = cv2.convexityDefects(max_contour, hull)

            if defects is not None and len(defects) > 5:
                return filled_external_area, "Asymmetric"
            else:
                return filled_external_area, "Symmetric"
        else:
            return threshold, "Symmetric"
    else:
        return threshold, "Symmetric"

def is_symmetric(image_path, mask_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img.shape[0] == 0 or img.shape[1] == 0:
            print(f"Error: Invalid image dimensions for {image_path}. Skipping.")
            return None
        _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print(f"Error: No contour found for {image_path}. Skipping.")
            return None

        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)

        # Ensure the bounding box dimensions are valid
        if w == 0 or h == 0:
            print(f"Error: Invalid bounding box dimensions for {image_path}. Skipping.")
            return None

        left_half = binary_image[y:y+h, x:int(x+w/2)].astype(np.float64)
        right_half = binary_image[y:y+h, int(x+w/2):x+w].astype(np.float64)

        if left_half.size == 0 or right_half.size == 0:
            print(f"Error: Invalid cropped halves for {image_path}. Skipping.")
            return None

        right_half_flipped = cv2.flip(right_half, 1)

        #Compare both halves
        difference = cv2.absdiff(left_half, right_half_flipped)
        similarity_score = np.sum(difference) / np.size(left_half)

        #print(f"Similarity Score: {similarity_score}")
        threshold = 12
        is_symmetric_result = similarity_score < threshold

        if is_symmetric_result:
            return "Symmetric"
        else:
            return "Asymmetric"

    except Exception as e:
        print(f"Error: {e}. Unable to process {image_path} or {mask_path}.")
        return None

def variance_whole(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    variance = np.var(img)

    return variance