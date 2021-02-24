"""
Generates a JSON file containing info from sliced DICOMs

JSON format:
"patient_id": {
		"roi_dir": "Root directory  containing the ROI slices",
		"axial": {
			"min_slice": min slices index containing info (not all background),
			"max_slice": max slices index containing info (not all background),
		},
		"coronal": { as above },
		"sagittal": { as above },
		"coordinates" : { contains minimum and maximum not blank informative pixel coordinate }
		"scan_dir": "Root directory containing the scan slices",
		"partition": if the patient belongs to train, validation or test set
	}

"""

import os
import re
import imageio
import numpy as np
import logging
import json
from utils.misc import remove_everything_after_last
# LOGGING
from utils import custom_logger
import logging

# define validation and test size
validation_size = 4
test_size = 4

# global
separator = '\\' # \\ windows, / unix
data_out_dir = '../data/out'
info_json = '../data/info.json'


def read_image_information_in_directory(directory):
	__files = [x for x in os.listdir(directory) if '.png' in x]
	__files.sort()
	# used to get the first and last informative slice
	bound = [None, None]
	# used to get the non background pixel coordinate in both direction, min and max
	min_info = [9999, 9999]
	max_info = [-1, -1]
	for i in range(len(__files) - 1):
		current_image_path = directory + '/' + __files[i]
		next_image_path = directory + '/' + __files[i + 1]
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
		bound[1] = len(__files) + 1
	return bound, min_info, max_info


if __name__ == "__main__":
	# read all directory in '...data/out'
	dir_names = []
	for root, dirs, files in os.walk(data_out_dir):
		if not dirs:
			dir_names += [os.path.abspath(root)]
	patient_map = {}
	for _dir in dir_names:
		patient_id = re.sub(r'^.*?data' + re.escape(separator) + 'out' + re.escape(separator), '', _dir).split(separator, 1)[0]
		if patient_id not in patient_map:
			patient_map[patient_id] = {}
			patient_map[patient_id]['coordinates'] = {}
		# ignore if cut directory
		if 'cut' in _dir:
			continue
		if 'roi' in _dir:
			# get information about informative images in 'roi' dir
			patient_map[patient_id]['roi_dir'] = remove_everything_after_last(_dir, separator)
			logging.info('Opening directory ' + _dir)
			info = read_image_information_in_directory(_dir)
			if 'axial' in _dir:
				# Y-X plane
				patient_map[patient_id]['axial'] = {}
				patient_map[patient_id]['axial']['min_slice'] = info[0][0]
				patient_map[patient_id]['axial']['max_slice'] = info[0][1]
				patient_map[patient_id]['coordinates']['min_y'] = info[1][0]
				patient_map[patient_id]['coordinates']['min_x'] = info[1][1]
				patient_map[patient_id]['coordinates']['max_y'] = info[2][0]
				patient_map[patient_id]['coordinates']['max_x'] = info[2][1]
			elif 'coronal' in _dir:
				# Z-X plane
				patient_map[patient_id]['coronal'] = {}
				patient_map[patient_id]['coronal']['min_slice'] = info[0][0]
				patient_map[patient_id]['coronal']['max_slice'] = info[0][1]
				patient_map[patient_id]['coordinates']['min_z'] = info[1][0]
				patient_map[patient_id]['coordinates']['max_z'] = info[2][0]
			elif 'sagittal' in _dir:
				patient_map[patient_id]['sagittal'] = {}
				patient_map[patient_id]['sagittal']['min_slice'] = info[0][0]
				patient_map[patient_id]['sagittal']['max_slice'] = info[0][1]
		else:
			patient_map[patient_id]['scan_dir'] = remove_everything_after_last(_dir, separator)
	# define to which set of data the patients belongs to (train, validation, test)
	total_patients = len(patient_map.keys())
	counter = 0
	for patient in patient_map:
		if counter < total_patients - test_size - validation_size:
			patient_map[patient]['partition'] = 'train'
		else:
			if total_patients - test_size - validation_size <= counter < total_patients - test_size:
				patient_map[patient]['partition'] = 'validation'
			else:
				patient_map[patient]['partition'] = 'test'
		counter = counter + 1
	logging.info("Writing JSON info file")
	with open(info_json, 'w') as outfile:
		json.dump(patient_map, outfile, indent=4)
	exit(0)
