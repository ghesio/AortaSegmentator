import json
import numpy as np
# LOGGING
from utils import custom_logger
from utils.misc import calculate_iou_score
import logging
import os
import csv

# load bin files for a given timestamp
timestamp = '020210210_152825'
bin_files = ['..//checkpoints//' + x for x in os.listdir('..//checkpoints//') if timestamp in x and '.npy' in x]
if not bin_files:
    exit(-1)
bin_files.sort()
# bin_files[0] = axial
# bin_files[1] = combined
# bin_files[2] = coronal
# bin_files[3] = roi
# bin_files[4] = sagittal
# define threshold interval and delta
threshold_interval = [0.6, 0.9]
delta = 0.05
# maps to store the IoU scores
iou_map_axial = {}
iou_map_coronal = {}
iou_map_sagittal = {}
iou_map_combined = {}
# vectors to load the data
array_axial = np.load(bin_files[0])
array_coronal = np.load(bin_files[2])
array_sagittal = np.load(bin_files[4])
array_combined = np.load(bin_files[1])
roi = np.load(bin_files[3])
logging.info('Loaded files ' + str(bin_files))
thresholds = np.arange(threshold_interval[0], threshold_interval[1], delta)
# iterate and calculate IoU score
for threshold in thresholds:
    threshold = round(threshold, 2)
    iou_map_axial[threshold] = []
    iou_map_coronal[threshold] = []
    iou_map_sagittal[threshold] = []
    iou_map_combined[threshold] = []
    logging.info("Evaluating threshold " + str(threshold))
    for i in range(array_axial.shape[0]):
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
