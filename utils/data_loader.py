import json
import numpy as np
import imageio
from utils import data_augmentation as da
from sklearn.model_selection import train_test_split

def get_data_set(direction, samples_from_each_patient=20, ratio_train = 0.8, ratio_val = 0.1, ratio_test = 0.1):
    if direction is not 'axial' or direction is not 'axial' or direction is not 'axial':
        return -1
    # define max and min intervals for data augmentation
    shift_array = np.array(np.arange(-5, 1 + 5, 1))
    shift_array = shift_array[shift_array != 0]
    rotation_array = shift_array
    zoom_array = np.array([0.8, 0.9, 1.1, 1.2, 1.3])
    with open('../data/info.json') as f:
        patient_map = json.load(f)
    # instantiate data and ground truth array
    scan_array = []
    roi_array = []
    # iterate through every patient in map
    for patient in patient_map:
        # get location of cut slices (both scan and roi)
        scan_cut_dir = patient_map[patient]['scan_cut_dir'] + '\\' + direction
        roi_cut_dir = patient_map[patient]['roi_cut_dir'] + '\\' + direction
        # get min and max informative slice indexes
        min_slice_index = patient_map[patient][direction]['min_slice']
        max_slice_index = patient_map[patient][direction]['max_slice']
        # draw random samples from the slices to load them
        random_indexes = np.random.randint(min_slice_index, max_slice_index, samples_from_each_patient)
        scan_slices = []
        roi_slices = []
        # load the slices from disk
        for index in random_indexes:
            scan_uri = scan_cut_dir + '\\' + direction + '_' + str.zfill(str(index), 4) + '.png'
            roi_uri = roi_cut_dir + '\\' + direction + '_' + str.zfill(str(index), 4) + '.png'
            scan_slices.append(np.array(imageio.imread(uri=scan_uri), dtype='uint8'))
            roi_slices.append(np.array(imageio.imread(uri=roi_uri), dtype='uint8'))
        assert len(roi_slices) == samples_from_each_patient == len(scan_slices)
        # iterate through data to be added to the dataset arrays
        for i in range(len(scan_slices)):
            current_slice = scan_slices[i]
            current_roi = roi_slices[i]
            # add to data set withut augmentation
            scan_array.append(current_slice)
            roi_array.append(current_roi)
            # augment using zoom
            for zoom in zoom_array:
                scan_array.append(da.zoom_image(current_slice, zoom))
                roi_array.append(current_roi)
            # augment using rotation
            for angle in rotation_array:
                scan_array.append(da.rotate_image(current_slice, angle))
                roi_array.append(current_roi)
            # augment using horizontal shift
            for shift in shift_array:
                scan_array.append(da.shift_image(current_slice, shift, axis=1))
                roi_array.append(current_roi)
            # augment using vertical shift
            for shift in shift_array:
                scan_array.append(da.shift_image(current_slice, shift, axis=0))
                roi_array.append(current_roi)
    # split in train-test-validation
    x_remaining, x_test, y_remaining, y_test = train_test_split(scan_array, roi_array, random_state=42,
                                                                test_size=ratio_test)
    # adjusts val ratio, w.r.t. remaining dataset
    ratio_remaining = 1 - ratio_test
    ratio_val_adjusted = ratio_val / ratio_remaining

    # Produces train and val splits.
    x_train, x_val, y_train, y_val = train_test_split(
        x_remaining, y_remaining, random_state=42, test_size=ratio_val_adjusted)
    return x_train, x_test, x_val, y_train, y_test, y_val


ret = get_data_set("axial")
print(len(ret[0]), len(ret[1]), len(ret[2]), len(ret[3]), len(ret[4]), len(ret[5]))
exit(0)
