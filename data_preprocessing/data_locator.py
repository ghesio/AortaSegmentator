"""
Generates a JSON file containing info from sliced DICOMs

JSON format:
"patient_id": {
		"roi_dir": "Root directory  containing the ROI slices",
		"axial": {
			"min_slice": min slices index containing info (not all background),
			"max_slice": max slices index containing info (not all background),
			"min_pixel_0": min non background pixel along rows of view,
			"min_pixel_1": min non background pixel along columns of view,
			"max_pixel_0": max non background pixel along rows of view,
			"max_pixel_1": min non background pixel along rows of view
		},
		"coronal": { as above },
		"sagittal": { as above },
		"scan_dir": "Root directory containing the scan slices"
	}

"""

import os
import re
import imageio
import numpy as np
import logging
import json

# Set logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("../locator.log"),
        logging.StreamHandler()
    ]
)


def read_image_information_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        if files:
            # used to get the first and last informative slice
            bound = [None, None]
            # used to get the non background pixel coordinate in both direction, min and max
            min_info = [9999, 9999]
            max_info = [-1, -1]
            for i in range(len(files) - 1):
                current_image_path = root + '\\' + files[i]
                next_image_path = root + '\\' + files[i + 1]
                # read the image into a numpy array
                current_image = np.array(imageio.imread(uri=current_image_path), dtype='uint8')
                next_image = np.array(imageio.imread(uri=next_image_path), dtype='uint8')
                background_color = current_image[0, 0]
                # a slice is informative if it's not only background
                if not np.all(current_image == background_color):
                    if bound[0] is None:
                        bound[0] = i + 1
                    if not np.all(current_image == background_color) and np.all(next_image == background_color):
                        bound[1] = i + 1
                    for j in range(current_image.shape[0]):
                        for k in range(current_image.shape[1]):
                            pixel = current_image[j][k]
                            if pixel != background_color:
                                if j < min_info[0]:
                                    min_info[0] = j
                                if k < min_info[1]:
                                    min_info[1] = k
                                if j > max_info[0]:
                                    max_info[0] = j
                                if k > max_info[1]:
                                    max_info[1] = k
            if bound[1] is None:
                bound[1] = len(files) + 1
            return bound, min_info, max_info


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


# read all directory in '...data/out'
dir_names = []
for root, dirs, files in os.walk('../data/out'):
    if not dirs:
        dir_names += [os.path.abspath(root)]
patient_map = {}
for dir in dir_names:
    patient_id = re.sub(r"^.*?data\\out\\", '', dir).split("\\", 1)[0]
    if patient_id not in patient_map:
        patient_map[patient_id] = {}
    # get information about informative images in 'roi' dir
    if 'roi' in dir:
        patient_map[patient_id]['roi_dir'] = remove_everything_after_last(dir)
        logging.info('Opening directory ' + dir)
        info = read_image_information_in_directory(dir)
        if 'axial' in dir:
            # Y-X plane
            patient_map[patient_id]['axial'] = {}
            patient_map[patient_id]['axial']['min_slice'] = info[0][0]
            patient_map[patient_id]['axial']['max_slice'] = info[0][1]
            patient_map[patient_id]['axial']['min_y'] = info[1][0]
            patient_map[patient_id]['axial']['min_x'] = info[1][1]
            patient_map[patient_id]['axial']['max_y'] = info[2][0]
            patient_map[patient_id]['axial']['max_x'] = info[2][1]
        elif 'coronal' in dir:
            # Z-X plane
            patient_map[patient_id]['coronal'] = {}
            patient_map[patient_id]['coronal']['min_slice'] = info[0][0]
            patient_map[patient_id]['coronal']['max_slice'] = info[0][1]
            patient_map[patient_id]['coronal']['min_z'] = info[1][0]
            patient_map[patient_id]['coronal']['min_x'] = info[1][1]
            patient_map[patient_id]['coronal']['max_z'] = info[2][0]
            patient_map[patient_id]['coronal']['max_x'] = info[2][1]
        elif 'sagittal' in dir:
            patient_map[patient_id]['sagittal'] = {}
            patient_map[patient_id]['sagittal']['min_slice'] = info[0][0]
            patient_map[patient_id]['sagittal']['max_slice'] = info[0][1]
            patient_map[patient_id]['sagittal']['min_z'] = info[1][0]
            patient_map[patient_id]['sagittal']['min_x'] = info[1][1]
            patient_map[patient_id]['sagittal']['max_y'] = info[2][0]
            patient_map[patient_id]['sagittal']['max_y'] = info[2][1]
    else:
        patient_map[patient_id]['scan_dir'] = remove_everything_after_last(dir)

logging.info("Writing JSON info file")
with open('../data/info.json', 'w') as outfile:
    json.dump(patient_map, outfile, indent=4)
exit(0)
