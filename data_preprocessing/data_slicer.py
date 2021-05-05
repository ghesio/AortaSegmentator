# https://simpleitk.readthedocs.io
# https://dicom.innolitics.com/ciods/ct-image/image-plane/00280030
# http://dicomiseasy.blogspot.com/2013/06/getting-oriented-using-image-plane.html

"""
Slices DICOM series file in data/in/patient_id_1, data/in/patient_id_2, ... into
data/out/patient_id_1/, data/in/patient_id_2 with all views
"""

import os
import sys
from utils.dicom_utils import convert_image_to_numpy_array, save_slices, convert_img, preprocess_slice
# LOGGING
from utils import custom_logger
import logging

data_in_dir = 'data/in'


def convert_dicom(patient):
    logging.info('Processing patient ' + patient)
    # input directories
    roi_dir = patient + 'roi'
    scan_dir = patient + 'scan'
    # output directories
    out_roi_dir = roi_dir.replace('in', 'out')
    out_scan_dir = scan_dir.replace('in', 'out')

    roi_array = convert_image_to_numpy_array(roi_dir, roi=True)
    scan_array = convert_image_to_numpy_array(scan_dir)

    #if roi_array.shape != scan_array.shape:
    #    logging.error('Mimatching shape for patient ' + patient)
    #    return

    # preprocess every slice
    for i in range(roi_array.shape[0]):
        scan_array[i, :, :] = preprocess_slice(scan_array[i, :, :], roi_array[i, :, :])

    # Save scan slices
    # AXIAL
    save_slices('axial', scan_array, out_scan_dir)
    # CORONAL
    save_slices('coronal', scan_array, out_scan_dir)
    # SAGITTAL
    save_slices('sagittal', scan_array, out_scan_dir)

    # Save scan slices
    # AXIAL
    save_slices('axial', roi_array, out_roi_dir)
    # CORONAL
    save_slices('coronal', roi_array, out_roi_dir)
    # SAGITTAL
    save_slices('sagittal', roi_array, out_roi_dir)


def main():
    directory_list = [data_in_dir + '/' + d + '/' for d in os.listdir(data_in_dir) if os.path.isdir(os.path.join(data_in_dir, d))]
    if not directory_list:
        logging.error("Nothing to process - aborting.")
        exit(-1)
    counter = 0
    for patient in directory_list:
        logging.info('Patient ' + str(counter) + '/' + str(len(directory_list)))
        convert_dicom(patient)
        counter = counter + 1
    exit(0)


if __name__=='__main__':
    main()
