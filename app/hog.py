import cv2
import numpy as np

# Load the image as grayscale
def gethog_64(img_gray):
    x = (128,128)
    new_img= cv2.resize(img_gray, x, interpolation= cv2.INTER_AREA)
    win_size = new_img.shape
    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = (8, 8)
    num_bins = 9


    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
    hog_descriptor = hog.compute(new_img)
    return hog_descriptor