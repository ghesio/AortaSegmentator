import json
import numpy as np
import imageio
from utils import image_augmentation as da
from sklearn.model_selection import train_test_split

# define max and min intervals for data augmentation
shift_array = np.array(np.arange(-5, 1 + 5, 1))
shift_array = shift_array[shift_array != 0]
rotation_array = shift_array
zoom_array = np.array([0.8, 0.9, 1.1, 1.2, 1.3])


def get_data_set(direction, samples_from_each_patient=20, ratio_val=0.1, ratio_test=0.1, normalization=True,
                 augmentation=True):
    """
    Load the whole dataset used to train the network
    :param direction: choose which direction (axial, coronal, sagittal)
    :param samples_from_each_patient: the number of samples to be drawn from all patients
    :param ratio_val: validation set ratio
    :param ratio_test: test set ration
    :param normalization: normalize between 0 and 1 using a min max scaler
    :param augmentation: if True augment data using rotation, zoom and shifts
    :return: the splitted dataset or -1 if a wrong direction is chose
    """
    if direction is not 'axial' and direction is not 'coronal' and direction is not 'sagittal':
        return -1
    with open('../data/info.json') as f:
        patient_map = json.load(f)
    # instantiate data and ground truth array
    scan_array = []
    roi_array = []
    # sort json
    sorted_patients = sorted(patient_map)
    sorted_map = {}
    for patient in sorted_patients:
        sorted_map[patient] = patient_map[patient]
    # iterate through every patient in sorted map
    for patient in sorted_map:
        # get location of cut slices (both scan and roi)
        scan_cut_dir = patient_map[patient]['scan_cut_dir'] + '\\' + direction
        roi_cut_dir = patient_map[patient]['roi_cut_dir'] + '\\' + direction
        # get min and max informative slice indexes
        min_slice_index = patient_map[patient][direction]['min_slice']
        max_slice_index = patient_map[patient][direction]['max_slice']
        # draw random samples from the slices to load them
        random_indexes = np.random.randint(min_slice_index, max_slice_index - 1, samples_from_each_patient)
        # load the slices from disk
        scan_slices = []
        roi_slices = []
        for index in random_indexes:
            scan_uri = scan_cut_dir + '\\' + direction + '_' + str.zfill(str(index), 4) + '.png'
            roi_uri = roi_cut_dir + '\\' + direction + '_' + str.zfill(str(index), 4) + '.png'
            scan_load = np.array(imageio.imread(uri=scan_uri), dtype='uint8')
            scan_slices.append(scan_load)
            roi_load = np.array(imageio.imread(uri=roi_uri), dtype='uint8')
            roi_slices.append(roi_load)
        assert len(roi_slices) == samples_from_each_patient == len(scan_slices)
        # iterate through data to be added to the dataset arrays
        for i in range(len(scan_slices)):
            current_slice = scan_slices[i]
            current_roi = roi_slices[i]
            # add to data set without augmentation
            scan_array.append(current_slice)
            roi_array.append(current_roi)
            if augmentation:
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
    if normalization:
        # a simple min-max scaling with predefined range (due to int_8 files)
        for i in range(len(scan_array)):
            scan_array[i] = scan_array[i] / 255
            roi_array[i] = roi_array[i] / 255
    # split in train-test-validation
    x_remaining, x_test, y_remaining, y_test = train_test_split(scan_array, roi_array, random_state=42,
                                                                test_size=ratio_test)
    # adjusts val ratio, w.r.t. remaining dataset
    ratio_remaining = 1 - ratio_test
    ratio_val_adjusted = ratio_val / ratio_remaining
    # produces train and val splits
    x_train, x_val, y_train, y_val = train_test_split(
        x_remaining, y_remaining, random_state=42, test_size=ratio_val_adjusted)
    # return splitted dataset
    return np.array(x_train, dtype='float32'), np.array(x_test, dtype='float32'), np.array(x_val, dtype='float32'), \
           np.array(y_train, dtype='float32'), np.array(y_test, dtype='float32'), np.array(y_val, dtype='float32')


def get_train_validation(direction, samples_from_each_patient=20, ratio_val=0.1, normalization=True, test_size=10,
                         augmentation=True):
    """
    Load train and validation from data
    :param direction: choose which direction (axial, coronal, sagittal)
    :param samples_from_each_patient: the number of samples to be drawn from all patients
    :param ratio_val: validation set ratio
    :param test_size: test set size (# of patients)
    :param normalization: normalize between 0 and 1 using a min max scaler
    :param augmentation: if True augment data using rotation, zoom and shifts
    :return: the splitted dataset or -1 if a wrong direction is chose
    """
    if direction is not 'axial' and direction is not 'coronal' and direction is not 'sagittal':
        return -1
    with open('../data/info.json') as f:
        patient_map = json.load(f)
    # instantiate data and ground truth array
    scan_array = []
    roi_array = []
    # sort json
    sorted_patients = sorted(patient_map)
    if len(sorted_patients) < test_size:
        return -1
    sorted_map = {}
    i = 0
    for patient in sorted_patients:
        if len(sorted_patients) - test_size == i:
            break
        sorted_map[patient] = patient_map[patient]
        i = i + 1
    # iterate through every patient in sorted map
    for patient in sorted_map:
        # get location of cut slices (both scan and roi)
        scan_cut_dir = patient_map[patient]['scan_cut_dir'] + '\\' + direction
        roi_cut_dir = patient_map[patient]['roi_cut_dir'] + '\\' + direction
        # get min and max informative slice indexes
        min_slice_index = patient_map[patient][direction]['min_slice']
        max_slice_index = patient_map[patient][direction]['max_slice']
        # draw random samples from the slices to load them
        random_indexes = np.random.randint(min_slice_index, max_slice_index - 1, samples_from_each_patient)
        # load the slices from disk
        scan_slices = []
        roi_slices = []
        for index in random_indexes:
            scan_uri = scan_cut_dir + '\\' + direction + '_' + str.zfill(str(index), 4) + '.png'
            roi_uri = roi_cut_dir + '\\' + direction + '_' + str.zfill(str(index), 4) + '.png'
            scan_load = np.array(imageio.imread(uri=scan_uri), dtype='uint8')
            scan_slices.append(scan_load)
            roi_load = np.array(imageio.imread(uri=roi_uri), dtype='uint8')
            roi_slices.append(roi_load)
        assert len(roi_slices) == samples_from_each_patient == len(scan_slices)
        # iterate through data to be added to the dataset arrays
        for i in range(len(scan_slices)):
            current_slice = scan_slices[i]
            current_roi = roi_slices[i]
            # add to data set without augmentation
            scan_array.append(current_slice)
            roi_array.append(current_roi)
            if augmentation:
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
    if normalization:
        # a simple min-max scaling with predefined range (due to int_8 files)
        for i in range(len(scan_array)):
            scan_array[i] = scan_array[i] / 255
            roi_array[i] = roi_array[i] / 255
    # split in train-test-validation
    x_train, x_val, y_train, y_val = train_test_split(scan_array, roi_array, random_state=42, test_size=ratio_val)
    # return splitted dataset
    return np.array(x_train, dtype='float32'), np.array(x_val, dtype='float32'), \
           np.array(y_train, dtype='float32'), np.array(y_val, dtype='float32')


def get_test(direction, samples_from_each_patient=20, test_size=10, normalization=True, augmentation=True):
    """
    Load the dataset used to train the network
    :param direction: choose which direction (axial, coronal, sagittal)
    :param samples_from_each_patient: the number of samples to be drawn from all patients
    :param test_size: test set size (# of patients)
    :param normalization: normalize between 0 and 1 using a min max scaler
    :param augmentation: if True augment data using rotation, zoom and shifts
    :return: the splitted dataset or -1 if a wrong direction is chose
    """
    if direction is not 'axial' and direction is not 'coronal' and direction is not 'sagittal':
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
    # sort json
    sorted_patients = sorted(patient_map)
    if len(sorted_patients) < test_size:
        return -1
    sorted_map = {}
    i = 0
    for patient in sorted_patients:
        sorted_map[patient] = patient_map[patient]
        if len(sorted_patients) - test_size <= i:
            sorted_map[patient]["test"] = True
        else:
            sorted_map[patient]["test"] = False
        i = i + 1
    # iterate through every patient in sorted map
    for patient in sorted_map:
        if sorted_map[patient]["test"] is False:
            continue
        # get location of cut slices (both scan and roi)
        scan_cut_dir = patient_map[patient]['scan_cut_dir'] + '\\' + direction
        roi_cut_dir = patient_map[patient]['roi_cut_dir'] + '\\' + direction
        # get min and max informative slice indexes
        min_slice_index = patient_map[patient][direction]['min_slice']
        max_slice_index = patient_map[patient][direction]['max_slice']
        # draw random samples from the slices to load them
        random_indexes = np.random.randint(min_slice_index, max_slice_index - 1, samples_from_each_patient)
        # load the slices from disk
        scan_slices = []
        roi_slices = []
        for index in random_indexes:
            scan_uri = scan_cut_dir + '\\' + direction + '_' + str.zfill(str(index), 4) + '.png'
            roi_uri = roi_cut_dir + '\\' + direction + '_' + str.zfill(str(index), 4) + '.png'
            scan_load = np.array(imageio.imread(uri=scan_uri), dtype='uint8')
            scan_slices.append(scan_load)
            roi_load = np.array(imageio.imread(uri=roi_uri), dtype='uint8')
            roi_slices.append(roi_load)
        assert len(roi_slices) == samples_from_each_patient == len(scan_slices)
        # iterate through data to be added to the dataset arrays
        for i in range(len(scan_slices)):
            current_slice = scan_slices[i]
            current_roi = roi_slices[i]
            # add to data set without augmentation
            scan_array.append(current_slice)
            roi_array.append(current_roi)
            if augmentation:
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
    if normalization:
        # a simple min-max scaling with predefined range (due to int_8 files)
        for i in range(len(scan_array)):
            scan_array[i] = scan_array[i] / 255
            roi_array[i] = roi_array[i] / 255
    # return test set
    return np.array(scan_array, dtype='float32'), np.array(roi_array, dtype='float32')


if __name__ == "__main__":
    whole = get_data_set("axial", augmentation=True, samples_from_each_patient=20)
    print(whole[0].shape, whole[1].shape, whole[2].shape, whole[3].shape, whole[4].shape, whole[5].shape)
    assert whole[0].shape == whole[3].shape and whole[1].shape == whole[4].shape and whole[2].shape == whole[5].shape
    train_val = get_train_validation("axial", augmentation=True, samples_from_each_patient=20, ratio_val=0.1, test_size=1)
    print(train_val[0].shape, train_val[1].shape)
    assert train_val[0].shape[1] == train_val[1].shape[1] and train_val[0].shape[2] == train_val[1].shape[2]
    test = get_test("axial", augmentation=True, samples_from_each_patient=20, test_size=1)
    print(test[0].shape, test[1].shape)
    assert test[0].shape == test[1].shape
    assert train_val[0].shape[0] + train_val[1].shape[0] + test[0].shape[0] == \
           whole[0].shape[0] + whole[1].shape[0] + whole[2].shape[0]
    exit(0)
