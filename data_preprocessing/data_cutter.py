import json
# LOGGING
import logging
from utils import custom_logger
import math
import os
import cv2
import imageio
import numpy as np

# data dirs
info_path = 'data/info.json'
data_out_path = 'data/slices/'
directions = ['axial', 'coronal', 'sagittal']
separator = "/"

def cut(directory, direction, partition, min_index, max_index):
    dir_files = [x for x in os.listdir(directory + separator + direction) if '.png' in x]
    dir_files.sort()
    root_dir = data_out_path + partition + separator + direction + separator
    if 'roi' in directory:
        out_dir = root_dir + 'labels' + separator
    else:
        out_dir = root_dir + 'scans' + separator
    increment = 0
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        already_cut = [x for x in os.listdir(out_dir) if '.png' in x]
        if already_cut:
            already_cut.sort()
            increment = int(already_cut[len(already_cut) - 1][-12:-4]) + 1
    k = 0
    for j in np.arange(min_index, max_index - 1):
        current_image_path = directory + separator + direction + separator + dir_files[j]
        out_image_path = out_dir + direction + '_' + str(k + increment).zfill(8) + '.png'
        # read the image into a numpy array
        current_image = np.array(imageio.imread(uri=current_image_path), dtype='uint8')
        cut_image = None
        if direction == 'axial':
            cut_image = current_image[lower_y:upper_y, lower_x:upper_x]
            cut_image = pad_to_new_shape(cut_image, (new_side_y, new_side_x))
        elif direction == 'coronal':
            pass  # ? the instruction below elif is skipped ?
            cut_image = current_image[lower_z:upper_z, lower_x:upper_x]
            cut_image = pad_to_new_shape(cut_image, (new_side_z, new_side_x))
        else:
            pass  # ? the instruction below elif is skipped ?
            cut_image = current_image[lower_z:upper_z, lower_y:upper_y]
            cut_image = pad_to_new_shape(cut_image, (new_side_z, new_side_y))
        logging.debug('Saving image to ' + out_image_path)
        if cut_image is None:
            logger.error('Nothing to save - ' + out_image_path)
        status = cv2.imwrite(filename=out_image_path, img=cut_image)
        if status is False:
            logging.error('Error saving image. Path:' + out_image_path
                          + " - image shape: " + str(cut_image.shape))
        k = k + 1


def pad_to_new_shape(already_cut, new_shape):
    # first axis
    first_sub = new_shape[0] - already_cut.shape[0]
    lower = 0
    upper = 0
    if first_sub != 0:
        # need to pad
        if first_sub % 2 != 0:
            lower = int(first_sub / 2) + 1
            upper = int(first_sub / 2)
        else:
            lower = int(first_sub / 2)
            upper = int(first_sub / 2)
        already_cut = np.pad(already_cut, ((lower, upper), (0, 0)), mode='constant', constant_values=0)
    second_sub = new_shape[1] - already_cut.shape[1]
    if second_sub != 0:
        if second_sub % 2 != 0:
            lower = int(second_sub / 2) + 1
            upper = int(second_sub / 2)
        else:
            lower = int(second_sub / 2)
            upper = int(second_sub / 2)
        already_cut = np.pad(already_cut, ((0, 0), (lower, upper)), mode='constant', constant_values=0)
    return already_cut


if __name__ == "__main__":

    just_check = False
    overwrite = True
    # read JSON containing information
    with open(info_path) as f:
        patient_map = json.load(f)
    # iterate through every patient and get bounding box vertexes
    x_coordinates_min = []
    x_coordinates_max = []
    y_coordinates_min = []
    y_coordinates_max = []
    z_coordinates_min = []
    z_coordinates_max = []
    try:
        for patient in patient_map:
            x_coordinates_min.append(patient_map[patient]['coordinates']['min_x'])
            x_coordinates_max.append(patient_map[patient]['coordinates']['max_x'])
            y_coordinates_min.append(patient_map[patient]['coordinates']['min_y'])
            y_coordinates_max.append(patient_map[patient]['coordinates']['max_y'])
            z_coordinates_min.append(patient_map[patient]['coordinates']['min_z'])
            z_coordinates_max.append(patient_map[patient]['coordinates']['max_z'])
    except KeyError:
        logging.error('Missing key in patient ' + patient)
        exit(-1)
    min_x = min(x_coordinates_min)
    max_x = max(x_coordinates_max)
    min_y = min(y_coordinates_min)
    max_y = max(y_coordinates_max)
    min_z = min(z_coordinates_min)
    max_z = max(z_coordinates_max)
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
    lower_y = center_y - int(new_side_y / 2)
    upper_y = center_y + int(new_side_y / 2)
    lower_x = center_x - int(new_side_x / 2)
    upper_x = center_x + int(new_side_x / 2)
    lower_z = center_z - int(new_side_z / 2)
    upper_z = center_z + int(new_side_z / 2)
    for patient in patient_map:
        scan_slices_dir = patient_map[patient]["scan_dir"]
        roi_slices_dir = patient_map[patient]["roi_dir"]
        for direction in directions:
            min_index = patient_map[patient][direction]["min_slice"]
            max_index = patient_map[patient][direction]["max_slice"]
            # check that we have the same number of slices in both scan and roi dirs to avoid dirt in dataset
            scan_files_list = [scan_slices_dir + separator + direction + separator + direction + '_' + str(x).zfill(4)
                          + '.png' for x in np.arange(min_index, max_index - 1)]
            roi_files_list = [roi_slices_dir + separator + direction + separator + direction + '_' + str(x).zfill(4)
                          + '.png' for x in np.arange(min_index, max_index - 1)]
            if all([os.path.isfile(f) for f in scan_files_list]) is False \
                or all([os.path.isfile(f) for f in roi_files_list]) is False:
                logging.error('Skipping patient - ' + patient + ' - missing slices in ' + direction)
                continue
            logging.info('Processing patient ' + patient + ' - direction ' + direction)
            # actually cut the slices
            cut(scan_slices_dir, direction, patient_map[patient]["partition"],
                min_index, max_index)
            cut(roi_slices_dir, direction, patient_map[patient]["partition"],
                min_index, max_index)
    exit(0)
