import json
import numpy as np
import imageio
from utils import image_augmentation as da
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# define max and min intervals for data augmentation
shift_array = np.array(np.arange(-5, 1 + 5, 1))
shift_array = shift_array[shift_array != 0]
rotation_array = np.copy(shift_array)
zoom_array = np.array([0.8, 0.9, 1.1, 1.2, 1.3])
# data file
data_file = 'data/info.json'


def get_train_set(direction, samples_from_each_patient=20, normalization=True, augmentation=True):
    """
    Load train set from data
    :param direction: choose which direction (axial, coronal, sagittal)
    :param samples_from_each_patient: the number of samples to be drawn from patients (0 for every slices)
    :param normalization: normalize between 0 and 1 using a min max scaler
    :param augmentation: if True augment data using rotation, zoom and shifts
    :return: data or -1 if a wrong direction is chose
    """
    if direction is not 'axial' and direction is not 'coronal' and direction is not 'sagittal':
        return -1
    with open(data_file) as f:
        patient_map = json.load(f)
    # instantiate data and ground truth array
    scan_array = []
    roi_array = []
    # iterate through every patient in the map
    for patient in patient_map:
        if patient_map[patient]['partition'] != 'train':
            continue
        # get location of cut slices (both scan and roi)
        scan_cut_dir = patient_map[patient]['scan_cut_dir'] + '\\' + direction
        roi_cut_dir = patient_map[patient]['roi_cut_dir'] + '\\' + direction
        # get min and max informative slice indexes
        min_slice_index = patient_map[patient][direction]['min_slice']
        max_slice_index = patient_map[patient][direction]['max_slice']
        # draw random samples or all slices
        if samples_from_each_patient == 0:
            drawn_indexes = np.arange(min_slice_index, max_slice_index - 1)
        else:
            drawn_indexes = np.random.randint(min_slice_index, max_slice_index - 1, samples_from_each_patient)
        # load the slices from disk
        scan_slices = []
        roi_slices = []
        for index in drawn_indexes:
            scan_uri = scan_cut_dir + '\\' + direction + '_' + str.zfill(str(index), 4) + '.png'
            roi_uri = roi_cut_dir + '\\' + direction + '_' + str.zfill(str(index), 4) + '.png'
            scan_load = np.array(imageio.imread(uri=scan_uri), dtype='uint8')
            scan_slices.append(scan_load)
            roi_load = np.array(imageio.imread(uri=roi_uri), dtype='uint8')
            roi_slices.append(roi_load)
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
    # return train set
    return shuffle(np.array(scan_array, dtype='float32'), np.array(roi_array, dtype='float32'), random_state=42)


def get_test_set(direction, samples_from_each_patient=0, normalization=True):
    """
    Load test set from data
    :param direction: choose which direction (axial, coronal, sagittal)
    :param samples_from_each_patient: the number of samples to be drawn from all patients
    :param normalization: normalize between 0 and 1 using a min max scaler
    :return: fata or -1 if a wrong direction is chose
    """
    if direction is not 'axial' and direction is not 'coronal' and direction is not 'sagittal':
        return -1
    with open(data_file) as f:
        patient_map = json.load(f)
    # instantiate data and ground truth array
    scan_array = []
    roi_array = []
    # iterate through every patient in the map
    for patient in patient_map:
        if patient_map[patient]['partition'] != 'test':
            continue
        # get location of cut slices (both scan and roi)
        scan_cut_dir = patient_map[patient]['scan_cut_dir'] + '\\' + direction
        roi_cut_dir = patient_map[patient]['roi_cut_dir'] + '\\' + direction
        # get min and max informative slice indexes
        min_slice_index = patient_map[patient][direction]['min_slice']
        max_slice_index = patient_map[patient][direction]['max_slice']
        # draw random samples or all slices
        if samples_from_each_patient == 0:
            drawn_indexes = np.arange(min_slice_index, max_slice_index - 1)
        else:
            drawn_indexes = np.random.randint(min_slice_index, max_slice_index - 1, samples_from_each_patient)
        # load the slices from disk
        scan_slices = []
        roi_slices = []
        for index in drawn_indexes:
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
    if normalization:
        # a simple min-max scaling with predefined range (due to int_8 files)
        for i in range(len(scan_array)):
            scan_array[i] = scan_array[i] / 255
            roi_array[i] = roi_array[i] / 255
    # return test set
    return np.array(scan_array, dtype='float32'), np.array(roi_array, dtype='float32')


def get_validation_set(direction, samples_from_each_patient=0, normalization=True):
    """
    Load validation set from data
    :param direction: choose which direction (axial, coronal, sagittal)
    :param samples_from_each_patient: the number of samples to be drawn from all patients
    :param normalization: normalize between 0 and 1 using a min max scaler
    :return: data or -1 if a wrong direction is chose
    """
    if direction is not 'axial' and direction is not 'coronal' and direction is not 'sagittal':
        return -1
    with open(data_file) as f:
        patient_map = json.load(f)
    # instantiate data and ground truth array
    scan_array = []
    roi_array = []
    # iterate through every patient in the map
    for patient in patient_map:
        if patient_map[patient]['partition'] != 'validation':
            continue
        # get location of cut slices (both scan and roi)
        scan_cut_dir = patient_map[patient]['scan_cut_dir'] + '\\' + direction
        roi_cut_dir = patient_map[patient]['roi_cut_dir'] + '\\' + direction
        # get min and max informative slice indexes
        min_slice_index = patient_map[patient][direction]['min_slice']
        max_slice_index = patient_map[patient][direction]['max_slice']
        # draw random samples or all slices
        if samples_from_each_patient == 0:
            drawn_indexes = np.arange(min_slice_index, max_slice_index - 1)
        else:
            drawn_indexes = np.random.randint(min_slice_index, max_slice_index - 1, samples_from_each_patient)
        # load the slices from disk
        scan_slices = []
        roi_slices = []
        for index in drawn_indexes:
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
    if normalization:
        # a simple min-max scaling with predefined range (due to int_8 files)
        for i in range(len(scan_array)):
            scan_array[i] = scan_array[i] / 255
            roi_array[i] = roi_array[i] / 255
    # return test set
    return np.array(scan_array, dtype='float32'), np.array(roi_array, dtype='float32')


def get_test_set_directories():
    with open(data_file) as f:
        patient_map = json.load(f)
        directories = []
        for patient in patient_map:
            if patient_map[patient]['partition'] == 'test':
                directories.append((patient_map[patient]['scan_dir'], patient_map[patient]['roi_dir']))
        return directories


if __name__ == "__main__":
    train = get_train_set('axial', samples_from_each_patient=20, augmentation=True)
    print(train[0].shape, train[1].shape)
    test = get_test_set('axial', samples_from_each_patient=20)
    print(test[0].shape, test[1].shape)
    validation = get_validation_set('axial', samples_from_each_patient=20)
    print(validation[0].shape, validation[1].shape)
    exit(0)
