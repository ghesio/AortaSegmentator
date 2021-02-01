#https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions

import numpy as np
from scipy.ndimage import rotate
from scipy.misc import face
import cv2
from matplotlib import pyplot as plt


def rotate_image(img, angle):
    """
    Return a rotated image using a random angle
    :param img: the image to be rotated
    :param angle: the angle
    :return: the rotated image
    """
    return rotate(img, angle, reshape=False)


def zoom_image(img, zoom_factor):
    """
    Return a zoomed image using a random zoom
    :param img: the image to be zoomed
    :param zoom_factor: the zoom factor
    :return: the zoomed image
    """
    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    # Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1, x1, y2, x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    return result


def shift_image(img, shift, axis=1):
    """
    Shift an image horizontally or vertically using a random value
    :param img: the image to be shifted
    :param shift: the shift factor
    :param axis: the axis for the switch, 1 for horizontal, 0 for vertical
    :return: the shifted image
    """
    img = np.roll(img, shift, axis)
    if shift > 0:
        if axis is 1:
            img[:, :shift] = 0
        else:
            img[:shift, :] = 0
    elif shift < 0:
        if axis is 1:
            img[:, shift:] = 0
        else:
            img[shift:, :] = 0
    return img


def test():
    """
    Test method
    :return: void
    """
    img = face()
    rot = rotate_image(img, 10)
    zoomed = zoom_image(img, 50)
    horizontal = shift_image(img, -300, axis=1)
    vertical = shift_image(img, 300, axis=0)
    fig, ax = plt.subplots(2, 3)
    ax[0][0].imshow(img)
    ax[0][1].imshow(rot)
    ax[1][0].imshow(zoomed)
    ax[1][1].imshow(horizontal)
    ax[1][2].imshow(vertical)

    plt.show()


#test()
#exit(0)
