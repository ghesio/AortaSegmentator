# https://simpleitk.readthedocs.io
# https://dicom.innolitics.com/ciods/ct-image/image-plane/00280030
# http://dicomiseasy.blogspot.com/2013/06/getting-oriented-using-image-plane.html

"""
Slices DICOM series file in data/in/patient_id_1, data/in/patient_id_2, ... into
data/out/patient_id_1/, data/in/patient_id_2 with all views and eventual downsampled views
"""

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
        logging.FileHandler("../slicer.log"),
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


def resample_image(itk_image, out_spacing=(1.0, 1.0, 1.0)):
    """
    Resample itk_image to new out_spacing
    :param itk_image: the input image
    :param out_spacing: the desired spacing
    :return: the resampled image
    """
    # get original spacing and size
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    # calculate new size
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]
    # instantiate resample filter with properties and execute it
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    return resample.Execute(itk_image)


def convert_dicom(input_dir, output_dir, downsample_factor=2, downsample_direction=(0, 0, 0),
                  directions=(1, 1, 1), equalization=True):
    """
    Write all dicom series slices into the output directory
    :param input_dir: the input dir containing DICOM series
    :param output_dir: the output dir which will contain the output slices
    :param downsample_factor: if specified also downsample the image
    :param downsample_direction: if specified the factor also downsample in the direction
    :param directions: which direction to get
    :param equalization: set to True to equalize the image
    :return: void
    """
    logging.info('Opening directory ' + input_dir)
    # Instantiate a DICOM reader
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(input_dir)
    reader.SetFileNames(dicom_names)
    # Execute the reader
    image = reader.Execute()
    logging.info('Image series read')
    # Log image size and spacing info BEFORE resample
    size = image.GetSize()
    logging.info('Image size - before resample (px): ' + str(size[0]) + 'x' + str(size[1]) + 'x' + str(size[2]))
    spacing = image.GetSpacing()
    logging.info('Spacing info - before resample (mm): ' + str(spacing[0]) + 'x' + str(spacing[1]) + 'x' +
                 str(spacing[2]))
    # Resample the image to get 1 px = 1 mm
    image = resample_image(image)
    # Log image size and spacing info after resample to check everything went good
    size = image.GetSize()
    logging.info('Image size - after resample (px): ' + str(size[0]) + 'x' + str(size[1]) + 'x' + str(size[2]))
    spacing = image.GetSpacing()
    logging.info('Spacing info - after resample (mm): ' + str(spacing[0]) + 'x' + str(spacing[1]) + 'x' +
                 str(spacing[2]))
    # Instantiate and execute orientation filter setting to RAI, to guarantee coherence
    #       across all DICOM images generating slices
    # RAI means
    #   X: patient Right to left
    #   Y: patient Anterior to posterior
    #   Z: patient Inferior to superior
    orientation_filter = sitk.DICOMOrientImageFilter()
    orientation_filter.SetDesiredCoordinateOrientation(DesiredCoordinateOrientation='RAI')
    image = orientation_filter.Execute(image)
    # Convert to numpy array
    # CARE as this operation goes from (x,y,z) to (z,)
    # Equalization of the image, this may be slow
    if equalization and 'roi' not in input_dir:
        logging.info('Equalization of the image')
        sitk.AdaptiveHistogramEqualization(image)
    image_array = sitk.GetArrayFromImage(image)
    if 'roi' in input_dir:
        # set B/W the image
        background_color = image_array[0][0][0]
        image_array[image_array == background_color] = 0
        image_array[image_array != 0] = 255
    # Calculate downsampled image
    image_array_downsampled = image_array[::downsample_factor, ::downsample_factor, ::downsample_factor]

    #
    # Save axial view
    #
    if directions[0] == 1:
        downsample = True if downsample_factor and downsample_direction[0] == 1 else False
        if downsample:
            save_directory_downsample = output_dir + '\\axial_downsample_' + str(downsample_factor) + '\\'
            if os.path.exists(save_directory_downsample):
                logging.warning('Deleting old directory ' + save_directory_downsample)
                shutil.rmtree(save_directory_downsample)
            os.makedirs(save_directory_downsample)
        save_directory = output_dir + '\\axial\\'
        if os.path.exists(save_directory):
            logging.warning('Deleting old directory ' + save_directory)
            shutil.rmtree(save_directory)
        os.makedirs(save_directory)
        # get max and min value for rescaling
        min = 99999
        max = -99999
        for i in range(image_array.shape[0]):
            if image_array[i, :, :].min() < min:
                min = image_array[i, :, :].min()
            if image_array[i, :, :].max() > max:
                max = image_array[i, :, :].max()
        logging.info('Rescaling (axial) - min ' + str(min) + ' max ' + str(max))
        logging.info('Start saving axial view in ' + save_directory)
        for i in range(image_array.shape[0]):
            output_file_name = save_directory + 'axial_' + str(i).zfill(4) + '.png'
            logging.debug('Saving image to ' + output_file_name)
            # TODO test more the image orientation filter to avoid rotating the images
            imageio.imwrite(output_file_name, convert_img(np.flipud(image_array[i, :, :]), min, max), format='png')
        if downsample:
            logging.info('Start saving axial (downlsampled) view in ' + save_directory_downsample)
            for i in range(image_array_downsampled.shape[0]):
                output_file_name = save_directory_downsample + 'axial_' + str(i).zfill(4) + '.png'
                logging.debug('Saving image to ' + output_file_name)
                imageio.imwrite(output_file_name, convert_img(np.flipud(image_array_downsampled[i, :, :]), min, max),
                                format='png')

    #
    # Save coronal view
    #
    if directions[1] == 1:
        downsample = True if downsample_factor and downsample_direction[1] == 1 else False
        if downsample:
            save_directory_downsample = output_dir + '\\coronal_downsample_' + str(downsample_factor) + '\\'
            if os.path.exists(save_directory_downsample):
                logging.warning('Deleting old directory ' + save_directory_downsample)
                shutil.rmtree(save_directory_downsample)
            os.makedirs(save_directory_downsample)
        save_directory = output_dir + '\\coronal\\'
        if os.path.exists(save_directory):
            logging.warning('Deleting old directory ' + save_directory)
            shutil.rmtree(save_directory)
        os.makedirs(save_directory)
        # get max and min value for rescaling
        min = 99999
        max = -99999
        for i in range(image_array.shape[1]):
            if image_array[:, i, :].min() < min:
                min = image_array[i, :, :].min()
            if image_array[:, i, :].max() > max:
                max = image_array[i, :, :].max()
        logging.info('Rescaling (coronal) - min ' + str(min) + ' max ' + str(max))
        logging.info('Start saving coronal view in ' + save_directory)
        for i in range(image_array.shape[1]):
            output_file_name = save_directory + 'coronal_' + str(i).zfill(4) + '.png'
            logging.debug('Saving image to ' + output_file_name)
            imageio.imwrite(output_file_name, convert_img(image_array[:, i, :], min, max), format='png')
        if downsample:
            logging.info('Start saving coronal (downlsampled) view in ' + save_directory_downsample)
            for i in range(image_array_downsampled.shape[1]):
                output_file_name = save_directory_downsample + 'coronal_' + str(i).zfill(4) + '.png'
                logging.debug('Saving image to ' + output_file_name)
                imageio.imwrite(output_file_name, convert_img(image_array_downsampled[:, i, :], min, max),
                                format='png')
    #
    # Save sagittal view
    #
    if directions[2] == 1:
        downsample = True if downsample_factor and downsample_direction[2] == 1 else False
        if downsample:
            save_directory_downsample = output_dir + '\\sagittal_downsample_' + str(downsample_factor) + '\\'
            if os.path.exists(save_directory_downsample):
                logging.warning('Deleting old directory ' + save_directory_downsample)
                shutil.rmtree(save_directory_downsample)
            os.makedirs(save_directory_downsample)
        save_directory = output_dir + '\\sagittal\\'
        if os.path.exists(save_directory):
            logging.warning('Deleting old directory ' + save_directory)
            shutil.rmtree(save_directory)
        os.makedirs(save_directory)
        # get max and min value for rescaling
        min = 99999
        max = -99999
        for i in range(image_array.shape[2]):
            if image_array[:, :, i].min() < min:
                min = image_array[:, :, i].min()
            if image_array[:, :, i].max() > max:
                max = image_array[:, :, i].max()
        logging.info('Rescaling (sagittal) - min ' + str(min) + ' max ' + str(max))
        logging.info('Start saving sagittal view in ' + save_directory)
        for i in range(image_array.shape[2]):
            output_file_name = save_directory + 'sagittal_' + str(i).zfill(4) + '.png'
            logging.debug('Saving image to ' + output_file_name)
            # FIXME test more the image orientation filter to avoid rotating the images
            imageio.imwrite(output_file_name, convert_img(np.fliplr(image_array[:, :, i]), min, max), format='png')
        if downsample:
            logging.info('Start saving sagittal (downlsampled) view in ' + save_directory_downsample)
            for i in range(image_array_downsampled.shape[2]):
                output_file_name = save_directory_downsample + 'sagittal_' + str(i).zfill(4) + '.png'
                logging.debug('Saving image to ' + output_file_name)
                imageio.imwrite(output_file_name, convert_img(np.fliplr(image_array_downsampled[:, :, i]), min, max),
                                format='png')


def convert_img(img, source_type_min=None, source_type_max=None, target_type_min=0, target_type_max=255,
                target_type=np.uint8):
    """
    Convert an image to another type for scaling, to avoid "Lossy conversion from ... to ..." problem
    :param img: An image
    :param source_type_min: the min value for rescaling (source)
    :param source_type_max: the max value for rescaling (source)
    :param target_type_min: the min value for rescaling (destination)
    :param target_type_max: the max value for rescaling (source)
    :param target_type: target data type
    :return: the rescaled image
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
    convert_dicom(key, value)
sys.exit(0)
