import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import math


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def extrapolate(lines, slope, y_top, y_bottom, b):
    x_avg = 0
    y_avg = 0
    for item in lines:
        x_avg = x_avg + item[0] + item[2]
        y_avg = y_avg + item[1] + item[3]
    x_avg = x_avg / (2 * len(lines))
    y_avg = y_avg / (2 * len(lines))

    x_top = int((y_top - b)/slope)
    x_bottom = int((y_bottom - b)/slope)
    return x_bottom, y_bottom, x_top, y_top



def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    left = []
    right = []

    slope_left = 0
    slope_right = 0
    y_top = 99999

    left_b_term = []
    right_b_term = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2-x1 < 0.01:
                continue
            slope = (float(y2-y1)/float(x2-x1))
            if slope > 0:
                right.append([x1, y1, x2, y2])
                slope_right = slope_right + slope
                b = y1 - slope * x1
                right_b_term.append(b)
            else:
                left.append([x1, y1, x2, y2])
                slope_left = slope_left + slope
                b = y1 - slope * x1
                left_b_term.append(b)
            y_top = min(y_top, y1, y2)

    if len(left) != 0:
        slope_left = slope_left / len(left)
        left_avg_b = sum(left_b_term) / len(left_b_term)
        x1, y1, x2, y2 = extrapolate(left, slope_left, y_top, img.shape[0], left_avg_b)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    if len(right) != 0:
        slope_right = slope_right / len(right)
        right_avg_b = sum(right_b_term) / len(right_b_term)
        x1, y1, x2, y2 = extrapolate(right, slope_right, y_top, img.shape[0], right_avg_b)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)


# def draw_line_advance(img, lines, color=[255, 0, 0], thickness=8):
#     left_x = []
#     right_x = []
#     left_y = []
#     right_y = []
#
#     slope_left = 0
#     slope_right = 0
#     y_top = 99999
#
#     left_b_term = []
#     right_b_term = []
#     for line in lines:
#         for x1, y1, x2, y2 in line:
#             slope = (float(y2-y1)/float(x2-x1))
#             if x1 > 0.66 * img.shape[1] or x2 > 0.66 * img.shape[1]:
#                 right_x.append(x1)
#                 right_y.append(y1)
#                 right_x.append(x2)
#                 right_y.append(y2)
#                 continue
#             if x1 < 0.33 * img.shape[1] or x2 < 0.33 * img.shape[1]:
#                 left_x.append(x1)
#                 left_y.append(y1)
#                 left_x.append(x2)
#                 left_y.append(y2)
#                 continue
#             if x2-x1 < 0.01:
#                 continue
#             y_top = min(y_top, y1, y2)
#             if slope > 0:
#                 right_x.append(x1)
#                 right_y.append(y1)
#                 right_x.append(x2)
#                 right_y.append(y2)
#             else:
#                 left_x.append(x1)
#                 left_y.append(y1)
#                 left_x.append(x2)
#                 left_y.append(y2)
#     z1 = np.polyfit(left_x, left_y, 3)
#     y_bottom =
#     cv2.polylines(img, )


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, thickness=8)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, alpha=0.7, beta=1., gamma=0.):
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


def detec_draw_line(img):
    gray_scale = grayscale(img)

    kernel_size = 5
    blur_gray = gaussian_blur(gray_scale, kernel_size)

    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    imshape = img.shape
    vertices = np.array([[(0, imshape[0]), (450, 320), (515, 320), (imshape[1], imshape[0])]], dtype=np.int32)

    masked_edges = region_of_interest(edges, vertices=vertices)

    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    line_img = weighted_img(line_img, img)

    return line_img



def image_test():
    # get the test image paths
    save_dir = 'test_images_output'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for item in os.listdir("test_images/"):
        test_image_path = os.path.join('test_images', item)
        image = mpimg.imread(test_image_path)
        temp = detec_draw_line(image)
        plt.imshow(temp)
        plt.show()
        mpimg.imsave(os.path.join(save_dir, item), temp)


def run():
    image_test()

if __name__ == '__main__':
    run()

