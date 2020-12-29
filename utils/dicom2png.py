import os
import sys
import SimpleITK as sitk
import numpy as np
import imageio
import logging
import math

# Set logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%dgit stat %H:%M:%S',
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


def convert_dicom(input_dir, output_dir, new_width=None, directions=[1, 1, 1], equalization=False):
    logging.info('Opening directory ' + input_dir)
    # Instantiate a DICOM reader
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(input_dir)
    reader.SetFileNames(dicom_names)
    # Execute the reader
    image = reader.Execute()
    logging.info('Image series read')
    if new_width:
        original_size = image.GetSize()
        original_spacing = image.GetSpacing()
        new_spacing = [(original_size[0] - 1) * original_spacing[0]
                       / (new_width - 1)] * 2
        new_size = [new_width, int((original_size[1] - 1) * original_spacing[1] / new_spacing[1])]
        image = sitk.Resample(image1=image, size=new_size,
                              transform=sitk.Transform(),
                              interpolator=sitk.sitkLinear,
                              outputOrigin=image.GetOrigin(),
                              outputSpacing=new_spacing,
                              outputDirection=image.GetDirection(),
                              defaultPixelValue=0,
                              outputPixelType=image.GetPixelID())
    # Equalization of the image, this is slow!
    if equalization:
        sitk.AdaptiveHistogramEqualization(image)
    size = image.GetSize()
    logging.info('Image size (px): ' + str(size[0]) + 'x' + str(size[1]) + 'x' + str(size[2]))
    # Convert to numpy array
    image_array = sitk.GetArrayFromImage(image)
    # Save axial view
    if directions[0] == 1:
        if new_width:
            axial_out_dir = output_dir + '\\axial_rescaled\\'
        else:
            axial_out_dir = output_dir + '\\axial\\'
        logging.info('Start saving axial view in ' + axial_out_dir)
        os.makedirs(axial_out_dir)
        for i in range(image_array.shape[0]):
            output_file_name = axial_out_dir + 'axial_' + str(i) + '.png'
            logging.debug('Saving image to ' + output_file_name)
            imageio.imwrite(output_file_name, convert_img(image_array[i, :, :]), format='png')
    # Save coronal view
    if directions[1] == 1:
        if new_width:
            coronal_out_dir = output_dir + '\\coronal_rescaled\\'
        else:
            coronal_out_dir = output_dir + '\\coronal\\'
        logging.info('Start saving coronal view in ' + coronal_out_dir)
        os.makedirs(coronal_out_dir)
        for i in range(image_array.shape[1]):
            output_file_name = coronal_out_dir + 'coronal_' + str(i) + '.png'
            logging.debug('Saving image to ' + output_file_name)
            imageio.imwrite(output_file_name, convert_img(image_array[:, i, :]), format='png')
    # Save sagittal view
    if directions[2] == 1:
        if new_width:
            sagittal_out_dir = output_dir + '\\sagittal_rescaled\\'
        else:
            sagittal_out_dir = output_dir + '\\sagittal\\'
        logging.info('Start saving sagittal view in ' + sagittal_out_dir)
        os.makedirs(sagittal_out_dir)
        for i in range(image_array.shape[2]):
            output_file_name = sagittal_out_dir + 'sagittal_' + str(i) + '.png'
            logging.debug('Saving image to ' + output_file_name)
            imageio.imwrite(output_file_name, convert_img(image_array[:, :, i]), format='png')


def convert_img(img, target_type_min=0, target_type_max=255, target_type=np.uint8):
    """
    Convert an image to another type for scaling, to avoid "Lossy conversion from ... to ..." problem
    :param img: the img
    :param target_type_min: the min target of scaling, default 0
    :param target_type_max: the max target of scaling, default 255
    :param target_type: the target type, default np.uint8
    :return: the converted image
    """
    imin = img.min()
    imax = img.max()
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
    convert_dicom(key, value)
    # Rescaled axial view
    convert_dicom(key, value, 128, [1, 0, 0])
