import pathlib

import cv2
import numpy as np


class ContourBasedSkySegmentation:
    def __init__(self) -> None:
        self.kernel = np.ones((3, 3), np.uint8)
        self.iterations = 4
        self.gaussian_kernel = (7, 7)

    def __call__(self, image_path: pathlib.Path) -> np.ndarray:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        padded = cv2.copyMakeBorder(thresh, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0)
        contours, _ = cv2.findContours(padded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
        # remove the padding
        biggest_contour = biggest_contour - 5
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.drawContours(mask, [biggest_contour], -1, (255, 255, 255), -1)
        blur = cv2.GaussianBlur(mask, self.gaussian_kernel, 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        opening = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, self.kernel, iterations=self.iterations
        )
        #
        return opening


class ColorBasedSkySegmentation:
    def __init__(self) -> None:
        self.kernel = np.ones((3, 3), np.uint8)
        self.iterations = 5
        self.gaussian_kernel = (9, 9)

    def __call__(self, image_path: pathlib.Path) -> np.ndarray:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # define range of sky blue color in HSV
        lower_blue = np.array([100])
        upper_blue = np.array([140])
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv[:, :, 0], lower_blue, upper_blue)
        # blur the mask
        blur = cv2.GaussianBlur(mask, self.gaussian_kernel, 0)
        # threshold again to obtain binary image by applying Otsu thresholding
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, self.kernel, iterations=self.iterations
        )
        mask[mask == 127] = 1
        return mask
