# https://simpleitk.readthedocs.io/en/master/link_DicomConvert_docs.html
# https://stackoverflow.com/questions/54160097/how-do-i-change-the-axis-simpleitkimageserieswriter-using

import os
import sys
import shutil
import SimpleITK as sitk
import numpy as np
import imageio
import logging

# Set logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("../info.log"),
        logging.StreamHandler()
    ]
)


def read_all_directories():
    """
    Read all last level subdirectories in '../data/in'
    :return: dictionary input-output directory (in ../data/out)
    """
    input_dir_names = []
    for root, dirs, files in os.walk('../data/in'):
        if not dirs:
            input_dir_names += [os.path.abspath(root)]
    file_map = {}
    for dir in input_dir_names:
        file_map[dir] = dir.replace('in', 'out')
    return file_map


def convert_dicom(input_dir, output_dir, downsample_factor=None, directions=[1, 1, 1], equalization=False):
    logging.info('Opening directory ' + input_dir)
    # Instantiate a DICOM reader
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(input_dir)
    reader.SetFileNames(dicom_names)
    # Execute the reader
    image = reader.Execute()
    logging.info('Image series read')
    # Equalization of the image, this is slow
    if equalization:
        logging.info('Equalization of the image')
        sitk.AdaptiveHistogramEqualization(image)
    size = image.GetSize()
    logging.info('Image size (px): ' + str(size[0]) + 'x' + str(size[1]) + 'x' + str(size[2]))
    # Convert to numpy array
    image_array = sitk.GetArrayFromImage(image)
    if downsample_factor:
        image_array = image_array[::downsample_factor, ::downsample_factor, ::downsample_factor]
    #
    # Save axial view
    #
    if directions[0] == 1:
        if downsample_factor:
            axial_out_dir = output_dir + '\\axial_rescaled\\'
        else:
            axial_out_dir = output_dir + '\\axial\\'
        logging.info('Start saving axial view in ' + axial_out_dir)
        if os.path.exists(axial_out_dir):
            shutil.rmtree(axial_out_dir)
        os.makedirs(axial_out_dir)
        # get max and min value for rescaling
        axial_min = 10000
        axial_max = -10000
        for i in range(image_array.shape[0]):
            if image_array[i, 0, 0].min() < axial_min:
                axial_min = image_array[i, :, :].min()
            if image_array[i, 0, 0].max() > axial_max:
                axial_max = image_array[i, :, :].max()
        logging.info('Rescaling (axial) - min ' + str(axial_min) + ' max ' + str(axial_max))
        for i in range(image_array.shape[0]):
            output_file_name = axial_out_dir + 'axial_' + str(i) + '.png'
            logging.debug('Saving image to ' + output_file_name)
            imageio.imwrite(output_file_name, convert_img(image_array[i, :, :], axial_min, axial_max), format='png')
    #
    # Save coronal view
    #
    if directions[1] == 1:
        if downsample_factor:
            coronal_out_dir = output_dir + '\\coronal_rescaled\\'
        else:
            coronal_out_dir = output_dir + '\\coronal\\'
        logging.info('Start saving coronal view in ' + coronal_out_dir)
        if os.path.exists(coronal_out_dir):
            shutil.rmtree(coronal_out_dir)
        os.makedirs(coronal_out_dir)
        # get max and min value for rescaling
        coronal_min = 10000
        coronal_max = -10000
        for i in range(image_array.shape[1]):
            if image_array[:, i, :].min() < coronal_min:
                coronal_min = image_array[:, i, :].min()
            if image_array[:, i, :].max() > coronal_max:
                coronal_max = image_array[:, i, :].max()
        logging.info('Rescaling (coronal) - min ' + str(coronal_min) + ' max ' + str(coronal_max))
        for i in range(image_array.shape[1]):
            output_file_name = coronal_out_dir + 'coronal_' + str(i) + '.png'
            logging.debug('Saving image to ' + output_file_name)
            imageio.imwrite(output_file_name, convert_img(image_array[:, i, :], coronal_min, coronal_max), format='png')
    #
    # Save sagittal view
    #
    if directions[2] == 1:
        if downsample_factor:
            sagittal_out_dir = output_dir + '\\sagittal_rescaled\\'
        else:
            sagittal_out_dir = output_dir + '\\sagittal\\'
        logging.info('Start saving sagittal view in ' + sagittal_out_dir)
        if os.path.exists(sagittal_out_dir):
            shutil.rmtree(sagittal_out_dir)
        os.makedirs(sagittal_out_dir)
        # get max and min value for rescaling
        sagittal_min = 10000
        sagittal_max = -10000
        for i in range(image_array.shape[2]):
            if image_array[:, :, i].min() < sagittal_min:
                sagittal_min = image_array[:, :, i].min()
            if image_array[:, :, i].max() > sagittal_max:
                sagittal_max = image_array[:, :, i].max()
        logging.info('Rescaling (sagittal) - min ' + str(sagittal_min) + ' max ' + str(sagittal_max))
        for i in range(image_array.shape[2]):
            output_file_name = sagittal_out_dir + 'sagittal_' + str(i) + '.png'
            logging.debug('Saving image to ' + output_file_name)
            imageio.imwrite(output_file_name, convert_img(image_array[:, :, i], sagittal_min, sagittal_max), format='png')


def convert_img(img, source_type_min=None, source_type_max=None, target_type_min=0, target_type_max=255, target_type=np.uint8):
    """
    Convert an image to another type for scaling, to avoid "Lossy conversion from ... to ..." problem
    :param img: the img
    :param target_type_min: the min target of scaling, default 0
    :param target_type_max: the max target of scaling, default 255
    :param target_type: the target type, default np.uint8
    :return: the converted image
    """
    if not source_type_min:
        imin = img.min()
    else:
        imin = source_type_min
    if not source_type_max:
        imax = img.max()
    else:
        imax = source_type_max
    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


file_map = read_all_directories()
if not file_map:
    logging.error("Nothing to process - aborting.")
    sys.exit(-1)

for key, value in file_map.items():
    # Original size
    convert_dicom(key, value, None, [1, 1, 1], False)
    # Rescaled axial view
    convert_dicom(key, value, 2, [1, 0, 0], True)
