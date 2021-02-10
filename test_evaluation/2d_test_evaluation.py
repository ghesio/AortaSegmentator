import json
from data_preprocessing.data_loader import get_test_set_directories
from utils.dicom_utils import convert_image_to_numpy_array
from utils.network_utils import get_pretrained_models, get_best_checkpoints
import keras
import numpy as np
import tensorflow as tf
# LOGGING
from utils import custom_logger
from utils.misc import calculate_iou_score
import logging
from datetime import datetime


# define the test size
test_size = 1
# get directories to load
directories = get_test_set_directories(test_size=test_size)
equalization = False
# load files for pickle name [0] = axial, [1]=coronal, [2]=sagittal, [3] = combined
best_files = get_best_checkpoints()
timestamp = datetime.today().strftime('%Y%m%d_%H%M%S')
files = [x[0:-11] + timestamp for x in best_files]
files.append(files[0].replace('axial', 'combined'))
files.append(files[0].replace('axial', 'roi'))
# load models [0] = axial, [1]=coronal, [2]=sagittal
models = get_pretrained_models()
# vectors to store the predictions on file system
array_axial = []
array_coronal = []
array_sagittal = []
array_combined = []
roi = []
# iterate all directories
for i in range(len(directories)):
    logging.info('Loading directories ' + str(directories[i]))
    scan_dir = str.replace(directories[i][0], 'out', 'in')
    roi_dir = str.replace(directories[i][1], 'out', 'in')
    # load scan image
    scan_array = convert_image_to_numpy_array(input_dir=scan_dir, equalization=equalization, padding=True, roi=False)
    roi.append(convert_image_to_numpy_array(input_dir=roi_dir, equalization=equalization, padding=True, roi=True))
    axial_shape = scan_array[0, :, :].shape
    coronal_shape = scan_array[:, 0, :].shape
    sagittal_shape = scan_array[:, :, 0].shape
    # get views
    prediction_axial = np.empty(shape=scan_array.shape)
    prediction_coronal = np.empty(shape=scan_array.shape)
    prediction_sagittal = np.empty(shape=scan_array.shape)
    prediction_combined = np.empty(shape=scan_array.shape)

    # predict axial value
    logging.info('Predicting axial values')
    for j in range(scan_array.shape[0]):
        current = tf.expand_dims(tf.expand_dims(np.flipud(scan_array[j, :, :]), axis=-1), axis=0)
        prediction_axial[j, :, :] = models[0].predict(current).reshape(axial_shape)
    # predict coronal value
    logging.info('Predicting coronal values')
    for j in range(scan_array.shape[1]):
        current = tf.expand_dims(tf.expand_dims(scan_array[:, j, :], axis=-1), axis=0)
        prediction_coronal[:, j, :] = models[1].predict(current).reshape(coronal_shape)
    logging.info('Predicting sagittal values')
    for j in range(scan_array.shape[2]):
        current = tf.expand_dims(tf.expand_dims(np.fliplr(scan_array[:, :, j]), axis=-1), axis=0)
        prediction_sagittal[:, :, j] = models[2].predict(current).reshape(sagittal_shape)
    # combine the views and calculate IoU
    prediction_combined = (prediction_axial + prediction_coronal + prediction_coronal) / 3.0
    array_axial.append(prediction_axial)
    array_coronal.append(prediction_coronal)
    array_sagittal.append(prediction_sagittal)
    array_combined.append(prediction_combined)
logging.info('Saving results')
np.save(files[0], array_axial)
np.save(files[1], array_coronal)
np.save(files[2], array_sagittal)
np.save(files[3], array_combined)
np.save(files[4], roi)

exit(0)

