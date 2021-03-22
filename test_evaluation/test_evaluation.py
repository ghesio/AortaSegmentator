import json
from utils.dicom_utils import convert_image_to_numpy_array
from utils.network_utils import get_pretrained_models, get_best_checkpoints
import keras
import numpy as np
import tensorflow as tf
# LOGGING
from utils import custom_logger
from utils.misc import calculate_iou_score
import logging
import csv
import os
from datetime import datetime


# define threshold interval and delta
threshold_interval = [0.6, 0.9]
delta = 0.05
# maps to store the IoU scores
iou_map_axial = {}
iou_map_coronal = {}
iou_map_sagittal = {}
iou_map_combined = {}

# get directories to load
directories = []
with open('data/info.json') as f:
    patient_map = json.load(f)
    for patient in patient_map:
        if patient_map[patient]['partition'] == 'test':
            directories.append((patient_map[patient]['scan_dir'], patient_map[patient]['roi_dir']))
equalization = True
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

    # TODO test if faster and it holds the same results by passing the whole array to predict(..)
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

thresholds = np.arange(threshold_interval[0], threshold_interval[1], delta)
# iterate and calculate IoU score
for threshold in thresholds:
    threshold = round(threshold, 2)
    iou_map_axial[threshold] = []
    iou_map_coronal[threshold] = []
    iou_map_sagittal[threshold] = []
    iou_map_combined[threshold] = []
    logging.info("Evaluating threshold " + str(threshold))
    for i in range(len(array_axial)):
        prediction_axial = np.copy(array_axial[i])
        prediction_coronal = np.copy(array_coronal[i])
        prediction_sagittal = np.copy(array_sagittal[i])
        prediction_combined = np.copy(array_combined[i])
        roi_array = roi[i]
        prediction_axial[prediction_axial >= threshold] = 1
        prediction_coronal[prediction_coronal >= threshold] = 1
        prediction_sagittal[prediction_sagittal >= threshold] = 1
        prediction_combined[prediction_combined >= threshold] = 1
        axial_iou_score = calculate_iou_score(prediction=prediction_axial, ground_truth=roi_array)
        iou_map_axial[threshold].append(axial_iou_score)
        coronal_iou_score = calculate_iou_score(prediction=prediction_coronal, ground_truth=roi_array)
        iou_map_coronal[threshold].append(coronal_iou_score)
        sagittal_iou_score = calculate_iou_score(prediction=prediction_sagittal, ground_truth=roi_array)
        iou_map_sagittal[threshold].append(sagittal_iou_score)
        combined_iou_score = calculate_iou_score(prediction=prediction_combined, ground_truth=roi_array)
        iou_map_combined[threshold].append(combined_iou_score)
        logging.debug('IoU scores (axial, coronal, sagittal, combined): ' + str(axial_iou_score) + ' '
                       + str(coronal_iou_score) + ' ' + str(sagittal_iou_score) + ' ' + str(combined_iou_score))

filename = '../data/results_' + timestamp + '.tsv'
logging.info('Saving tsv file - ' + filename)
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerow(['threshold', 'type', 'iou_scores', 'iou_mean', 'iou_std', 'iou_var'])
    # axial
    for threshold in iou_map_axial:
        writer.writerow([threshold, 'axial', str(iou_map_axial[threshold])[1:-1],
                         round(float(np.mean(iou_map_axial[threshold])), 4),
                         round(float(np.std(iou_map_axial[threshold])), 4),
                         round(float(np.var(iou_map_axial[threshold])), 4)])
    for threshold in iou_map_coronal:
        writer.writerow([threshold, 'coronal', str(iou_map_coronal[threshold])[1:-1],
                         round(float(np.mean(iou_map_coronal[threshold])), 4),
                         round(float(np.std(iou_map_coronal[threshold])), 4),
                         round(float(np.var(iou_map_coronal[threshold])), 4)])
    for threshold in iou_map_sagittal:
        writer.writerow([threshold, 'sagittal', str(iou_map_sagittal[threshold])[1:-1],
                         round(float(np.mean(iou_map_sagittal[threshold])), 4),
                         round(float(np.std(iou_map_sagittal[threshold])), 4),
                         round(float(np.var(iou_map_sagittal[threshold])), 4)])
    for threshold in iou_map_combined:
        writer.writerow([threshold, 'combined', str(iou_map_combined[threshold])[1:-1],
                         round(float(np.mean(iou_map_combined[threshold])), 4),
                         round(float(np.std(iou_map_combined[threshold])), 4),
                         round(float(np.var(iou_map_combined[threshold])), 4)])
exit(0)

