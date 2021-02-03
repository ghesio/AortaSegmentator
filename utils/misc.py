import numpy as np
from utils import custom_logger
import logging

def remove_everything_after_last(haystack, needle='\\', n=1):
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


def calculate_iou_score(prediction_a, prediction_b):
    # calculate intersection
    intersection = ((prediction_a == 1) & (prediction_b == 1)).sum()
    # calculate union
    union = (a == 1).sum() + (b == 1).sum() - intersection
    return intersection/union


if __name__ == "__main__":
    a = np.random.randint(2, size=(10, 10, 10))
    b = a
    assert calculate_iou_score(a, b) == 1.0
    exit(0)
