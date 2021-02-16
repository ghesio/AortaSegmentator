# https://simpleitk.readthedocs.io
# https://dicom.innolitics.com/ciods/ct-image/image-plane/00280030
# http://dicomiseasy.blogspot.com/2013/06/getting-oriented-using-image-plane.html

"""
Slices DICOM series file in data/in/patient_id_1, data/in/patient_id_2, ... into
data/out/patient_id_1/, data/in/patient_id_2 with all views
"""

import os
import sys
from utils.dicom_utils import convert_image_to_numpy_array, save_slices, convert_img
# LOGGING
from utils import custom_logger
import logging


def __read_all_directories():
    """
    Read all last level subdirectories in '../data/in'
    :return: dictionary input-output directory (in ../data/out)
    """
    input_dir_names = []
    for root, dirs, files in os.walk('data/in'):
        if not dirs:
            input_dir_names += [os.path.abspath(root)]
    __file_map = {}
    for __dir in input_dir_names:
        __file_map[__dir] = __dir.replace('in', 'out')
    return __file_map


def __convert_dicom(input_dir, out_root_dir, directions=(1, 1, 1), equalization=True):
    """
    Write all dicom series slices into the output directory
    :param input_dir: the input dir containing DICOM series
    :param out_root_dir: the output dir which will contain the output slices in sub folders
    :param directions: which direction to get
    :param equalization: set to True to equalize the image
    :return: void
    """
    logging.info('Opening directory ' + input_dir)
    if 'roi' in input_dir:
        image_array = convert_image_to_numpy_array(input_dir, equalization=equalization, roi=True)
    else:
        image_array = convert_image_to_numpy_array(input_dir, equalization=equalization)
    #
    # Save axial view
    #
    if directions[0] == 1:
        save_slices('axial', image_array, out_root_dir)
    #
    # Save coronal view
    #
    if directions[1] == 1:
        save_slices('coronal', image_array, out_root_dir)
    #
    # Save sagittal view
    #
    if directions[2] == 1:
        save_slices('sagittal', image_array, out_root_dir)


def main():
    file_map = __read_all_directories()
    if not file_map:
        logging.error("Nothing to process - aborting.")
        exit(-1)
    for key, value in file_map.items():
        # set equalization to false for faster testing
        __convert_dicom(key, value, equalization=True)
    exit(0)


if __name__ == "__main__":
    main()
