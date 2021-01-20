#https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions

import numpy as np
from scipy.ndimage import rotate
from scipy.misc import face
import cv2
from matplotlib import pyplot as plt


def random_rotations(img, angles=(-10, 10)):
    """
    Return a rotated image using a random angle
    :param img: the image to be rotated
    :param angles: the angle range
    :return: the rotated image
    """
    rotation = np.random.randint(angles[0], angles[1])
    return rotate(img, rotation, reshape=False)


def cv2_clipped_zoom(img, zoom=(0.85, 1.20)):
    """
    Return a zoomed image using a random zoom
    :param img: the image to be zoomed
    :param zoom: the zoom range
    :return: the zoomed image
    """
    zoom_factor = np.random.randint(zoom[0], zoom[1])
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


def shift_image(img, shift=(-10, 10), axis=1):
    """
    Shift an image horizontally or vertically using a random value
    :param img: the image to be shifted
    :param shift: the shift range
    :param axis: the axis for the switch, 1 for horizontal, 0 for vertical
    :return: the shifted image
    """
    delta = np.random.randint(shift[0], shift[1])
    img = np.roll(img, delta, axis)
    if delta > 0:
        if axis is 1:
            img[:, :delta] = 0
        else:
            img[:delta, :] = 0
    elif delta < 0:
        if axis is 1:
            img[:, delta:] = 0
        else:
            img[delta:, :] = 0
    return img


def test():
    """
    Test method
    :return: void
    """
    img = face()
    rot = random_rotations(img, [10, 30])
    zoom = cv2_clipped_zoom(img, [10, 20])
    horizontal = shift_image(img, [-300, 300], axis=1)
    vertical = shift_image(img, [-300, 300], axis=0)

    fig, ax = plt.subplots(2, 3)
    ax[0][0].imshow(img)
    ax[0][1].imshow(rot)
    ax[1][0].imshow(zoom)
    ax[1][1].imshow(horizontal)
    ax[1][2].imshow(vertical)

    plt.show()


test()
exit(0)
