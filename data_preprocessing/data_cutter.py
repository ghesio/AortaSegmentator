import json
import math
import os, shutil
import numpy as np
# LOGGING
from utils import custom_logger
import logging
import cv2
import imageio


def __cut(directory):
    dir_files = [x for x in os.listdir(directory) if '.png' in x]
    dir_files.sort()
    for j in range(len(dir_files)):
        current_image_path = directory + '/' + dir_files[j]
        if 'roi' in _dir:
            out_dir = directory.replace('roi', 'roi_cut')
        else:
            out_dir = directory.replace('scan', 'scan_cut')
        out_image_path = out_dir + '/' + dir_files[j]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # read the image into a numpy array
        current_image = np.array(imageio.imread(uri=current_image_path), dtype='uint8')
        cut_image = None
        if 'axial' in directory:
            cut_image = current_image[lower_y:upper_y, lower_x:upper_x]
            cut_image = pad_to_new_shape(cut_image, (new_side_y, new_side_x))
        elif 'coronal' in directory:
            pass  # ? the instruction below elif is skipped ?
            cut_image = current_image[lower_z:upper_z, lower_x:upper_x]
            cut_image = pad_to_new_shape(cut_image, (new_side_z, new_side_x))
        elif 'sagittal' in directory:
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
    lower = 0
    upper = 0
    if second_sub != 0:
        lower = int(second_sub / 2) + 1
        upper = int(second_sub / 2)
    else:
        lower = int(second_sub / 2)
        upper = int(second_sub / 2)
    already_cut = np.pad(already_cut, ((0, 0), (lower, upper)), mode='constant', constant_values=0)
    return already_cut


if __name__ == "__main__":
    # flags
    just_check = False
    overwrite = True
    # data dirs
    info_path = 'data/info.json'
    data_out_path = 'data/out/'
    # read JSON containing information
    with open(info_path) as f:
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
        if patient_map[patient]['coordinates']['min_y'] < min_y:
            min_y = patient_map[patient]['coordinates']['min_y']
        if patient_map[patient]['coordinates']['max_y'] > max_y:
            max_y = patient_map[patient]['coordinates']['max_y']
        if patient_map[patient]['coordinates']['min_x'] < min_x:
            min_x = patient_map[patient]['coordinates']['min_x']
        if patient_map[patient]['coordinates']['max_x'] > max_x:
            max_x = patient_map[patient]['coordinates']['max_x']
        if patient_map[patient]['coordinates']['min_z'] < min_z:
            min_z = patient_map[patient]['coordinates']['min_z']
        if patient_map[patient]['coordinates']['max_z'] > max_z:
            max_z = patient_map[patient]['coordinates']['max_z']
        patient_map[patient]['roi_cut_dir'] = patient_map[patient]['roi_dir'].replace('roi', 'roi_cut')
        patient_map[patient]['scan_cut_dir'] = patient_map[patient]['scan_dir'].replace('scan', 'scan_cut')
        if overwrite:
            if os.path.isdir(patient_map[patient]['roi_cut_dir']):
                shutil.rmtree(patient_map[patient]['roi_cut_dir'])
            if os.path.isdir(patient_map[patient]['scan_cut_dir']):
                shutil.rmtree(patient_map[patient]['scan_cut_dir'])
    logging.info("Updating JSON info file")
    with open(info_path, 'w') as outfile:
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
    # iterate through directories
    dir_names = [x[0] for x in os.walk(data_out_path) if ('axial' in x[0] or 'coronal' in x[0] or 'sagittal' in x[0])
                 and 'cut' not in x[0]]
    dir_names.sort()
    lower_y = center_y - int(new_side_y / 2)
    upper_y = center_y + int(new_side_y / 2)
    lower_x = center_x - int(new_side_x / 2)
    upper_x = center_x + int(new_side_x / 2)
    lower_z = center_z - int(new_side_z / 2)
    upper_z = center_z + int(new_side_z / 2)
    for _dir in dir_names:
        logging.info('Processing ' + _dir)
        if not just_check:
            __cut(_dir)
    # check integrity: for each original slice there must be a cut slice
    for patient in patient_map:
        # roi
        roi_cut_dir = patient_map[patient]['roi_cut_dir']
        roi_dir = patient_map[patient]['roi_dir']
        # axial
        roi_files = [x for x in os.listdir(roi_dir + '/axial/') if '.png' in x]
        roi_cut_files = [x for x in os.listdir(roi_cut_dir + '/axial/') if '.png' in x]
        difference = [x for x in roi_cut_files if x not in set(roi_files)]
        if difference:
            logging.error('Missing slices in directory ' + roi_cut_dir)
        # coronal
        roi_files = [x for x in os.listdir(roi_dir + '/coronal/') if '.png' in x]
        roi_cut_files = [x for x in os.listdir(roi_cut_dir + '/coronal/') if '.png' in x]
        difference = [x for x in roi_cut_files if x not in set(roi_files)]
        if difference:
            logging.error('Missing slices in directory ' + roi_cut_dir)
        # axial
        roi_files = [x for x in os.listdir(roi_dir + '/sagittal/') if '.png' in x]
        roi_cut_files = [x for x in os.listdir(roi_cut_dir + '/sagittal/') if '.png' in x]
        difference = [x for x in roi_cut_files if x not in set(roi_files)]
        if difference:
            logging.error('Missing slices in directory ' + roi_cut_dir)
        # scan
        scan_cut_dir = patient_map[patient]['roi_cut_dir']
        scan_dir = patient_map[patient]['roi_dir']
        # axial
        scan_files = [x for x in os.listdir(scan_dir + '/axial/') if '.png' in x]
        scan_cut_files = [x for x in os.listdir(scan_cut_dir + '/axial/') if '.png' in x]
        difference = [x for x in scan_cut_files if x not in set(scan_files)]
        if difference:
            logging.error('Missing slices in directory ' + scan_cut_dir)
        # coronal
        scan_files = [x for x in os.listdir(scan_dir + '/coronal/') if '.png' in x]
        scan_cut_files = [x for x in os.listdir(scan_cut_dir + '/coronal/') if '.png' in x]
        difference = [x for x in scan_cut_files if x not in set(scan_files)]
        if difference:
            logging.error('Missing slices in directory ' + scan_cut_dir)
        # axial
        scan_files = [x for x in os.listdir(scan_dir + '/sagittal/') if '.png' in x]
        scan_cut_files = [x for x in os.listdir(scan_cut_dir + '/sagittal/') if '.png' in x]
        difference = [x for x in scan_cut_files if x not in set(scan_files)]
        if difference:
            logging.error('Missing slices in directory ' + scan_cut_dir)
    exit(0)
