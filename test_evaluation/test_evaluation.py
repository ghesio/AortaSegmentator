import json
from utils.dicom_utils import convert_image_to_numpy_array, save_prediction_slices_with_scan, preprocess_slice, \
    postprocess_prediticion
from utils.network_utils import get_pretrained_models, get_best_checkpoints, get_preprocessor, backbone, architecture
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
import seaborn
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# define threshold interval and delta
threshold_interval = [0.3, 0.9]
delta = 0.02
# maps to store the IoU scores
iou_map_axial = {}
iou_map_coronal = {}
iou_map_sagittal = {}
iou_map_combined = {}

preprocessor = get_preprocessor()
# get best threshold and best_view
file_path = 'results/' + backbone + '_' + architecture + '/results_best.txt'
with open(file_path, 'r') as file:
    data = file.read().replace('\n', '')
splitted = data.split(' - ')
best_view = splitted[1]
best_threshold = float(splitted[2])


# get test and directories to load
test_directories = []
with open('data/info.json') as f:
    patient_map = json.load(f)
    for patient in patient_map:
        if patient_map[patient]['partition'] == 'test':
            test_directories.append((patient_map[patient]['scan_dir'], patient_map[patient]['roi_dir']))

# load models [0] = axial, [1]=coronal, [2]=sagittal
models = get_pretrained_models()
# iterate all directories in test set
k = 0
test_scores = []
test_scores_coherence = []
test_scores_majority = []

for i in range(len(test_directories)):
    logging.info('Loading directories ' + str(test_directories[i]))
    scan_dir = str.replace(test_directories[i][0], 'out', 'in')
    roi_dir = str.replace(test_directories[i][1], 'out', 'in')
    # load scan image
    scan_array = convert_image_to_numpy_array(input_dir=scan_dir)
    roi_array = convert_image_to_numpy_array(input_dir=roi_dir, roi=True)
    # preprocess every slice
    for ii in range(roi_array.shape[0]):
        scan_array[ii, :, :] = preprocess_slice(scan_array[ii, :, :])
    roi_array = roi_array / 255
    axial_shape = scan_array[0, :, :].shape
    coronal_shape = scan_array[:, 0, :].shape
    sagittal_shape = scan_array[:, :, 0].shape
    # get views
    prediction_axial = np.empty(shape=scan_array.shape)
    prediction_coronal = np.empty(shape=scan_array.shape)
    prediction_sagittal = np.empty(shape=scan_array.shape)
    prediction_combined = np.empty(shape=scan_array.shape)
    logging.info('Predicting axial values')
    os.makedirs('results/' + backbone + '_' + architecture + '/' + str(k) + '/heatmap/axial/')
    for j in range(scan_array.shape[0]):
        current = tf.expand_dims(tf.expand_dims(preprocessor(scan_array[j, :, :]), axis=-1), axis=0)
        prediction_axial[j, :, :] = models[0].predict(current).reshape(axial_shape)
        heatmap_file = 'results/' + backbone + '_' + architecture + '/' + str(k) + '/heatmap/axial/axial_' + str.zfill(str(j), 4) + '.png'
        sns_map = seaborn.heatmap(prediction_axial[j, :, :])
        plt.show(sns_map)
        plt.savefig(heatmap_file)
        plt.clf()
    logging.info('Predicting coronal values')
    os.makedirs('results/' + backbone + '_' + architecture + '/' + str(k) + '/heatmap/coronal/')
    for j in range(scan_array.shape[1]):
        current = tf.expand_dims(tf.expand_dims(preprocessor(scan_array[:, j, :]), axis=-1), axis=0)
        prediction_coronal[:, j, :] = models[1].predict(current).reshape(coronal_shape)
        heatmap_file = 'results/' + backbone + '_' + architecture + '/' \
                       + str(k) + '/heatmap/coronal/coronal_' + str.zfill(str(j), 4) + '.png'
        sns_map = seaborn.heatmap(prediction_coronal[:, j, :])
        plt.show(sns_map)
        plt.savefig(heatmap_file)
        plt.clf()
    logging.info('Predicting sagittal values')
    os.makedirs('results/' + backbone + '_' + architecture + '/' + str(k) + '/heatmap/sagittal/')
    for j in range(scan_array.shape[2]):
        current = tf.expand_dims(tf.expand_dims(preprocessor(scan_array[:, :, j]), axis=-1), axis=0)
        prediction_sagittal[:, :, j] = models[2].predict(current).reshape(sagittal_shape)
        heatmap_file = 'results/' + backbone + '_' + architecture + '/' \
                       + str(k) + '/heatmap/sagittal/sagittal_' + str.zfill(str(j), 4) + '.png'
        sns_map = seaborn.heatmap(prediction_sagittal[:, :, j])
        plt.show(sns_map)
        plt.savefig(heatmap_file)
        plt.clf()
    if best_view == 'axial':
        prediction_axial[prediction_axial >= best_threshold] = 1
        prediction_axial[prediction_axial != 1] = 0
        coherence = postprocess_prediticion(prediction_axial)
        majority = postprocess_prediticion(prediction_array=prediction_axial, majority_voting=True)
        test_scores.append(calculate_iou_score(prediction=prediction_axial, ground_truth=roi_array))
        test_scores_coherence.append(calculate_iou_score(prediction=coherence, ground_truth=roi_array))
        test_scores_majority.append(calculate_iou_score(prediction=majority, ground_truth=roi_array))

        save_prediction_slices_with_scan(best_direction=best_view, scan_array=scan_array, roi_array=roi_array,
                                         prediction=prediction_axial,
                                         root_dir='results/' + backbone + '_' + architecture + '/' + str(k))
        save_prediction_slices_with_scan(best_direction=best_view, scan_array=scan_array, roi_array=roi_array,
                                         prediction=coherence,
                                         root_dir='results/' + backbone + '_' + architecture + '/coherence/' + str(
                                             k))
        save_prediction_slices_with_scan(best_direction=best_view, scan_array=scan_array, roi_array=roi_array,
                                         prediction=majority,
                                         root_dir='results/' + backbone + '_' + architecture + '/majority/' + str(
                                             k))
    if best_view == 'coronal':
        prediction_coronal[prediction_coronal >= best_threshold] = 1
        prediction_coronal[prediction_coronal != 1] = 0
        coherence = postprocess_prediticion(prediction_coronal)
        majority = postprocess_prediticion(prediction_array=prediction_coronal, majority_voting=True)
        test_scores.append(calculate_iou_score(prediction=prediction_coronal, ground_truth=roi_array))
        test_scores_coherence.append(calculate_iou_score(prediction=coherence, ground_truth=roi_array))
        test_scores_majority.append(calculate_iou_score(prediction=majority, ground_truth=roi_array))

        save_prediction_slices_with_scan(best_direction=best_view, scan_array=scan_array, roi_array=roi_array,
                                         prediction=prediction_coronal,
                                         root_dir='results/' + backbone + '_' + architecture + '/' + str(k))
        save_prediction_slices_with_scan(best_direction=best_view, scan_array=scan_array, roi_array=roi_array,
                                         prediction=coherence,
                                         root_dir='results/' + backbone + '_' + architecture + '/coherence/' + str(
                                             k))
        save_prediction_slices_with_scan(best_direction=best_view, scan_array=scan_array, roi_array=roi_array,
                                         prediction=majority,
                                         root_dir='results/' + backbone + '_' + architecture + '/majority/' + str(
                                             k))
        if best_view == 'sagittal':
            prediction_sagittal[prediction_sagittal >= best_threshold] = 1
            prediction_sagittal[prediction_sagittal != 1] = 0
            coherence = postprocess_prediticion(prediction_sagittal)
            majority = postprocess_prediticion(prediction_array=prediction_sagittal, majority_voting=True)
            test_scores.append(calculate_iou_score(prediction=prediction_axial, ground_truth=roi_array))
            test_scores_coherence.append(calculate_iou_score(prediction=coherence, ground_truth=roi_array))
            test_scores_majority.append(calculate_iou_score(prediction=majority, ground_truth=roi_array))

            save_prediction_slices_with_scan(best_direction=best_view, scan_array=scan_array, roi_array=roi_array,
                                             prediction=prediction_sagittal,
                                             root_dir='results/' + backbone + '_' + architecture + '/' + str(k))
            save_prediction_slices_with_scan(best_direction=best_view, scan_array=scan_array, roi_array=roi_array,
                                             prediction=coherence,
                                             root_dir='results/' + backbone + '_' + architecture + '/coherence/' + str(
                                                 k))
            save_prediction_slices_with_scan(best_direction=best_view, scan_array=scan_array, roi_array=roi_array,
                                             prediction=majority,
                                             root_dir='results/' + backbone + '_' + architecture + '/majority/' + str(
                                                 k))
    if best_view == 'combined':
        # combine the views and calculate IoU
        prediction_combined = (prediction_axial + prediction_coronal + prediction_coronal) / 3.0
        prediction_combined[prediction_combined >= best_threshold] = 1
        prediction_combined[prediction_combined != 1] = 0
        coherence = postprocess_prediticion(prediction_combined)
        majority = postprocess_prediticion(prediction_array=prediction_combined, majority_voting=True)
        test_scores.append(calculate_iou_score(prediction=prediction_combined, ground_truth=roi_array))
        test_scores_coherence.append(calculate_iou_score(prediction=coherence, ground_truth=roi_array))
        test_scores_majority.append(calculate_iou_score(prediction=majority, ground_truth=roi_array))

        save_prediction_slices_with_scan(best_direction=best_view, scan_array=scan_array, roi_array=roi_array,
                                         prediction=prediction_combined,
                                         root_dir='results/' + backbone + '_' + architecture + '/' + str(k))
        save_prediction_slices_with_scan(best_direction=best_view, scan_array=scan_array, roi_array=roi_array,
                                         prediction=coherence,
                                         root_dir='results/' + backbone + '_' + architecture + '/coherence/' + str(
                                             k))
        save_prediction_slices_with_scan(best_direction=best_view, scan_array=scan_array, roi_array=roi_array,
                                         prediction=majority,
                                         root_dir='results/' + backbone + '_' + architecture + '/majority/' + str(
                                             k))
    k = k + 1
text_file = open('results/' + backbone + '_' + architecture + '/results_best.txt', 'a')
text_file.write('\r\nTest results: ' + ' '.join([str(score) for score in test_scores]))
text_file.write('\r\nTest results (coherence): ' + ' '.join([str(score) for score in test_scores_coherence]))
text_file.write('\r\nTest results (majority): ' + ' '.join([str(score) for score in test_scores_majority]))
text_file.write('\r\nTest average: ' + str(round(float(np.mean(test_scores)), 4)))
text_file.write('\r\nTest average (coherence): ' + str(round(float(np.mean(test_scores_coherence)), 4)))
text_file.write('\r\nTest average (majority): ' + str(round(float(np.mean(test_scores_majority)), 4)))

text_file.close()
exit(0)
