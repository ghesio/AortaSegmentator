# LOGGING
import logging
from utils import custom_logger
import numpy as np
import cv2

def remove_everything_after_last(haystack, needle='/', n=1):
    """
    Remove everything after the n instance of a char in a string
    :param haystack: the input string
    :param needle: what to search
    :param n: 1 last instance, 2 second-to-last, ecc.
    :return: the manipulated string
    """
    while n > 0:
        idx = haystack.rfind(needle)
        if idx >= 0:
            haystack = haystack[:idx]
            n -= 1
        else:
            break
    return haystack


def convert_img(img, source_type_min=None, source_type_max=None, target_type_min=0, target_type_max=255,
                target_type=np.uint8):
    """
    Convert an image to another type for scaling, to avoid "Lossy conversion from ... to ..." problem
    :param img: An image
    :param source_type_min: the min value for rescaling (source)
    :param source_type_max: the max value for rescaling (source)
    :param target_type_min: the min value for rescaling (destination)
    :param target_type_max: the max value for rescaling (destination)
    :param target_type: target data type
    :return: the rescaled image
    """
    if not source_type_min:
        input_min = img.min()
    else:
        input_min = source_type_min
    if not source_type_max:
        input_max = img.max()
    else:
        input_max = source_type_max
    logging.info('Rescaling - min ' + str(input_min) + ' max ' + str(input_max))
    a = (target_type_max - target_type_min) / (input_max - input_min)
    b = target_type_max - a * input_max
    new_img = (a * img + b).astype(target_type)
    return new_img


def calculate_iou_score(prediction, ground_truth):
    # calculate intersection
    intersection = ((prediction == 1) & (ground_truth == 1)).sum()
    # calculate union
    union = (prediction == 1).sum() + (ground_truth == 1).sum() - intersection
    if union == 0:
        return 0.0
    else:
        return intersection/union


def calculate_intersection_on_prediction(roi_slice, prediction_slice):
    # channel are in BGR order
    to_return = np.zeros(shape=(np.shape(roi_slice)[0], np.shape(roi_slice)[1], 3))
    intersection = cv2.bitwise_and(roi_slice/255, prediction_slice/255)
    union = cv2.bitwise_or(roi_slice/255, prediction_slice/255)
    left = union - prediction_slice/255
    right = union - roi_slice/255
    # set to white in intersection
    blue = intersection * 255
    green = intersection * 255
    red = intersection * 255
    # set to blue in left (pixel in roi but not in prediction)
    blue = blue + left * 255
    blue[blue == 510] = 255
    # set to red in right (pixel in prediction but not in roi)
    red = red + right * 255
    red[red == 510] = 255
    to_return[:, :, 0] = blue
    to_return[:, :, 1] = green
    to_return[:, :, 2] = red
    return to_return


def calculate_intersection_on_prediction_with_scan(scan_slice, roi_slice, prediction_slice):
    # channel are in BGR order
    to_return = cv2.merge((scan_slice, scan_slice, scan_slice))
    intersection = cv2.bitwise_and(roi_slice, prediction_slice)
    union = cv2.bitwise_or(roi_slice, prediction_slice)
    left = union - prediction_slice
    right = union - roi_slice
    # set to white in intersection
    blue = intersection
    green = intersection
    red = intersection
    # set to blue in left (pixel in roi but not in prediction)
    blue = blue + left
    blue[blue == 510] = 255
    # set to red in right (pixel in prediction but not in roi)
    red = red + right
    red[red == 510] = 255
    to_return[(blue == 255), 0] = 255
    to_return[(green == 255), 1] = 255
    to_return[(red == 255), 2] = 255
    return to_return


if __name__ == "__main__":
    roi = np.array(cv2.imread('E:\\Tesi\\AortaSegmentator\\data\\slices\\test\\axial\\labels\\axial_00000118.png', cv2.IMREAD_GRAYSCALE))
    prediction = np.array(cv2.imread('E:\\Tesi\\AortaSegmentator\\data\\slices\\test\\axial\\labels\\axial_00000131.png', cv2.IMREAD_GRAYSCALE))
    scan = np.array(cv2.imread('E:\\Tesi\\AortaSegmentator\\data\\slices\\test\\axial\\scans\\axial_00000118.png',cv2.IMREAD_GRAYSCALE))
    test = calculate_intersection_on_prediction_with_scan(scan, roi, prediction)
    cv2.imwrite(filename="C:\\Users\\ghesio\\Desktop\\test.png", img=test)
