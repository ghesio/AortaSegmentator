import json
from data_preprocessing.data_loader import get_test_set_directories
from utils.dicom_utils import convert_image_to_numpy_array
from utils.network_utils import get_model, get_pretrained_models
import keras
import numpy as np
import tensorflow as tf
# LOGGING
from utils import custom_logger
from utils.misc import calculate_iou_score
import logging

# define the test size
test_size = 1
# get directories to load
directories = get_test_set_directories(test_size=test_size)
equalization = False
# load models [0] = axial, [1]=coronal, [2]=sagittal
models = get_pretrained_models()
# define a threshold for which the prediction is good (or None)
threshold = 0.7
iou_array_axial = []
iou_array_coronal = []
iou_array_sagittal = []
iou_array_combined = []
# iterate all directories
for i in range(len(directories)):
    logging.info('Loading directories ' + str(directories[i]))
    scan_dir = str.replace(directories[i][0], 'out', 'in')
    roi_dir = str.replace(directories[i][1], 'out', 'in')
    # load scan image
    scan_array = convert_image_to_numpy_array(input_dir=scan_dir, equalization=equalization, padding=True, roi=False)
    roi_array = convert_image_to_numpy_array(input_dir=roi_dir, equalization=equalization, padding=True, roi=True)
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
    if threshold:
        prediction_axial[prediction_axial >= threshold] = 1
        prediction_coronal[prediction_coronal >= threshold] = 1
        prediction_sagittal[prediction_sagittal >= threshold] = 1
        prediction_combined[prediction_combined >= threshold] = 1
    axial_iou_score = calculate_iou_score(prediction=prediction_axial, ground_truth=roi_array)
    iou_array_axial.append(axial_iou_score)
    coronal_iou_score = calculate_iou_score(prediction=prediction_coronal, ground_truth=roi_array)
    iou_array_coronal.append(coronal_iou_score)
    sagittal_iou_score = calculate_iou_score(prediction=prediction_sagittal, ground_truth=roi_array)
    iou_array_sagittal.append(sagittal_iou_score)
    combined_iou_score = calculate_iou_score(prediction=prediction_combined, ground_truth=roi_array)
    iou_array_combined.append(combined_iou_score)
    logging.info('IoU scores (axial, coronal, sagittal, combined): ' + str(axial_iou_score) + ' '
                 + str(coronal_iou_score) + ' ' + str(sagittal_iou_score) + ' ' + str(combined_iou_score))

logging.info('IoU stats (mean, std deviation, variance')
logging.info('Axial' + ' ' + str(np.mean(iou_array_axial)) + ' ' + str(np.std(iou_array_axial)) + ' '
             + str(np.var(iou_array_axial)))
logging.info('Coronal' + ' ' + str(np.mean(iou_array_coronal)) + ' ' + str(np.std(iou_array_coronal)) + ' '
             + str(np.var(iou_array_coronal)))
logging.info('Sagittal' + ' ' + str(np.mean(iou_array_sagittal)) + ' ' + str(np.std(iou_array_sagittal)) + ' '
             + str(np.var(iou_array_sagittal)))
logging.info('Combined' + ' ' + str(np.mean(iou_array_combined)) + ' ' + str(np.std(iou_array_combined)) + ' '
             + str(np.var(iou_array_combined)))
exit(0)

