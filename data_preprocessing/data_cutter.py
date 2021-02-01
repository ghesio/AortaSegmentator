import json
import math
import os
import numpy as np
import imageio
# LOGGING
from utils import custom_logger
import logging


def __cut(directory):
    for __root, __dirs, __files in os.walk(directory):
        if __files:
            for __i in range(len(__files) - 1):
                current_image_path = __root + '\\' + __files[__i]
                if 'roi' in _dir:
                    out_dir = __root.replace('roi', 'roi_cut')
                else:
                    out_dir = __root.replace('scan', 'scan_cut')
                out_image_path = out_dir + '\\' + __files[__i]
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                # read the image into a numpy array
                current_image = np.array(imageio.imread(uri=current_image_path), dtype='uint8')
                cut_image = None
                lower_y = center_y - int(new_side_y / 2)
                upper_y = center_y + int(new_side_y / 2)
                lower_x = center_x - int(new_side_x / 2)
                upper_x = center_x + int(new_side_x / 2)
                lower_z = center_z - int(new_side_z / 2)
                upper_z = center_z + int(new_side_z / 2)

                if 'axial' in __root:
                    cut_image = current_image[lower_y:upper_y, lower_x:upper_x]
                    if cut_image.shape[0] < new_side_y:
                        pad_value = int((new_side_y - cut_image.shape[0]) / 2)
                        cut_image = np.pad(cut_image, ((pad_value + 1, pad_value), (0, 0)),
                                           mode='constant', constant_values=0)
                    if cut_image.shape[1] < new_side_x:
                        pad_value = int((new_side_x - cut_image.shape[1]) / 2)
                        cut_image = np.pad(cut_image, ((0, 0), (pad_value + 1, pad_value)),
                                           mode='constant', constant_values=0)
                if 'coronal' in __root:
                    cut_image = current_image[lower_z:upper_z, lower_x:upper_x]
                    if cut_image.shape[0] < new_side_z:
                        pad_value = int((new_side_z - cut_image.shape[0]) / 2)
                        cut_image = np.pad(cut_image, ((pad_value + 1, pad_value), (0, 0)),
                                           mode='constant', constant_values=0)
                    if cut_image.shape[1] < new_side_x:
                        pad_value = int((new_side_x - cut_image.shape[1]) / 2)
                        cut_image = np.pad(cut_image, ((0, 0), (pad_value + 1, pad_value)),
                                           mode='constant', constant_values=0)
                if 'sagittal' in __root:
                    cut_image = current_image[lower_z:upper_z, lower_y:upper_y]
                    if cut_image.shape[0] < new_side_z:
                        pad_value = int((new_side_z - cut_image.shape[0]) / 2)
                        cut_image = np.pad(cut_image, ((pad_value + 1, pad_value), (0, 0)),
                                           mode='constant', constant_values=0)
                    if cut_image.shape[1] < new_side_y:
                        pad_value = int((new_side_y - cut_image.shape[1]) / 2)
                        cut_image = np.pad(cut_image, ((0, 0), (pad_value + 1, pad_value)),
                                           mode='constant', constant_values=0)
                logging.debug('Saving image to ' + out_image_path)
                imageio.imwrite(out_image_path, cut_image, format='png')


if __name__ == "__main__":
    # dry run flag
    dry_run = False
    # read JSON containing information
    with open('../data/info.json') as f:
        patient_map = json.load(f)
    # initialize minima and maxima variables
    min_x = 99999
    max_x = -99999
    min_y = 99999
    max_y = -99999
    min_z = 99999
    max_z = -99999
    # iterate through every patient and get bounding box vertexes
    for patient in patient_map:
        if patient_map[patient]['axial']['min_y'] < min_y:
            min_y = patient_map[patient]['axial']['min_y']
        if patient_map[patient]['axial']['max_y'] > max_y:
            max_y = patient_map[patient]['axial']['max_y']
        if patient_map[patient]['axial']['min_x'] < min_x:
            min_x = patient_map[patient]['axial']['min_x']
        if patient_map[patient]['axial']['max_x'] > max_x:
            max_x = patient_map[patient]['axial']['max_x']
        if patient_map[patient]['coronal']['min_z'] < min_z:
            min_z = patient_map[patient]['coronal']['min_z']
        if patient_map[patient]['coronal']['max_z'] > max_z:
            max_z = patient_map[patient]['coronal']['max_z']
        patient_map[patient]['roi_cut_dir'] = patient_map[patient]['roi_dir'].replace('roi', 'roi_cut')
        patient_map[patient]['scan_cut_dir'] = patient_map[patient]['scan_dir'].replace('scan', 'scan_cut')

    logging.info("Updating JSON info file")
    with open('../data/info.json', 'w') as outfile:
        json.dump(patient_map, outfile, indent=4)

    logging.info("Buonding box location (before padding) (x,y,z): (" + str(min_x) + "-" + str(max_x) + ") x (" +
                 str(min_y) + "-" + str(max_y) + ") x (" + str(min_z) + "-" + str(max_z) + ")")

    # get max side value to pad to 32px multiple
    side_x = max_x - min_x
    side_y = max_y - min_y
    side_z = max_z - min_z
    # calculate center
    center_x = math.ceil((max_x + min_x) / 2)
    center_y = math.ceil((max_y + min_y) / 2)
    center_z = math.ceil((max_z + min_z) / 2)
    logging.info('Buonding box center: (x,y,z) ' + str(center_x) + 'x' + str(center_y) + 'x' + str(center_z))
    i = 1
    new_side_x = None
    new_side_y = None
    new_side_z = None
    while True:
        if side_x < 32 * i:
            new_side_x = int(32 * i)
            break
        else:
            i = i + 1
    i = 1
    while True:
        if side_y < 32 * i:
            new_side_y = int(32 * i)
            break
        else:
            i = i + 1
    i = 1
    while True:
        if side_z < 32 * i:
            new_side_z = int(32 * i)
            break
        else:
            i = i + 1
    logging.info(
        "Buonding box location (padded) (x,y,z): (" + str(int(center_x - new_side_x / 2)) + "-" + str(int(center_x +
            new_side_x / 2)) + ") x (" + str(int(center_y - new_side_y / 2)) + "-" + str(int(center_y +
            new_side_y / 2)) + ") x (" + str(int(center_z - new_side_z / 2)) + "-" + str(int(center_z +
            new_side_z / 2)) + ")")
    if dry_run:
        exit(1)
    # iterate through directories
    dir_names = []
    for root, dirs, files in os.walk('../data/out'):
        if not dirs:
            dir_names += [os.path.abspath(root)]
    for _dir in dir_names:
        if 'cut' in _dir:
            continue
        logging.info('Processing ' + _dir)
        __cut(_dir)
    exit(0)