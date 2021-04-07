import dicom_utils as du
from utils.network_utils import get_pretrained_models, get_best_checkpoints, get_preprocessor
import SimpleITK as sitk
import numpy as np
import tensorflow as tf


preprocessor = get_preprocessor()
models = get_pretrained_models()
threshold = 0.9

def predict(dicom_location, out_dir):
    # convert the dicom to image array
    image_array = du.convert_image_to_numpy_array(input_dir=dicom_location)
    # preprocess the image
    preprocessed_image = preprocessor(image_array)
    # get the shapes
    axial_shape = image_array[0, :, :].shape
    coronal_shape = image_array[:, 0, :].shape
    sagittal_shape = image_array[:, :, 0].shape
    # create empty array for views
    prediction_axial = np.empty(shape=image_array.shape)
    prediction_coronal = np.empty(shape=image_array.shape)
    prediction_sagittal = np.empty(shape=image_array.shape)
    prediction_combined = np.empty(shape=image_array.shape)
    # TODO test if faster and it holds the same results by passing the whole array to predict(..)
    # predict axial value
    print('Predicting axial values')
    for j in range(preprocessed_image.shape[0]):
        current = tf.expand_dims(tf.expand_dims(preprocessor(preprocessed_image[j, :, :]), axis=-1), axis=0)
        prediction_axial[j, :, :] = models[0].predict(current).reshape(axial_shape)
    # predict coronal value
    print('Predicting coronal values')
    for j in range(preprocessed_image.shape[1]):
        current = tf.expand_dims(tf.expand_dims(preprocessor(preprocessed_image[:, j, :]), axis=-1), axis=0)
        prediction_coronal[:, j, :] = models[1].predict(current).reshape(coronal_shape)
    # predict sagittal values
    print('Predicting sagittal values')
    for j in range(preprocessed_image.shape[2]):
        current = tf.expand_dims(tf.expand_dims(preprocessor(preprocessed_image[:, :, j]), axis=-1), axis=0)
        prediction_sagittal[:, :, j] = models[2].predict(current).reshape(sagittal_shape)
    # combine the views
    prediction_combined = (prediction_axial + prediction_coronal + prediction_coronal) / 3.0
    # de-normalize the prediction before saving
    prediction_axial[prediction_axial >= threshold] = 255
    prediction_axial[prediction_axial != 255] = 0
    prediction_coronal[prediction_coronal >= threshold] = 255
    prediction_coronal[prediction_coronal != 255] = 0
    prediction_sagittal[prediction_sagittal >= threshold] = 255
    prediction_sagittal[prediction_sagittal != 255] = 0
    prediction_combined[prediction_combined >= threshold] = 255
    prediction_combined[prediction_combined != 255] = 0
    # AXIAL
    image = sitk.GetImageFromArray(prediction_axial)
    image.SetSpacing((1.0, 1.0, 1.0))
    orientation_filter = sitk.DICOMOrientImageFilter()
    orientation_filter.SetDesiredCoordinateOrientation(DesiredCoordinateOrientation='RAI')
    image = orientation_filter.Execute(image)
    print('Saving axial prediction')
    sitk.WriteImage(image, out_dir + '/axial.nii')
    # coronal
    image = sitk.GetImageFromArray(prediction_coronal)
    image.SetSpacing((1.0, 1.0, 1.0))
    orientation_filter = sitk.DICOMOrientImageFilter()
    orientation_filter.SetDesiredCoordinateOrientation(DesiredCoordinateOrientation='RAI')
    image = orientation_filter.Execute(image)
    print('Saving coronal prediction')
    sitk.WriteImage(image, out_dir + '/coronal.nii')
    # sagittal
    image = sitk.GetImageFromArray(prediction_sagittal)
    image.SetSpacing((1.0, 1.0, 1.0))
    orientation_filter = sitk.DICOMOrientImageFilter()
    orientation_filter.SetDesiredCoordinateOrientation(DesiredCoordinateOrientation='RAI')
    image = orientation_filter.Execute(image)
    print('Saving sagittal prediction')
    sitk.WriteImage(image, out_dir + '/sagittal.nii')
    # combined
    print('Saving combined prediction')
    image = sitk.GetImageFromArray(prediction_combined)
    image.SetSpacing((1.0, 1.0, 1.0))
    orientation_filter = sitk.DICOMOrientImageFilter()
    orientation_filter.SetDesiredCoordinateOrientation(DesiredCoordinateOrientation='RAI')
    image = orientation_filter.Execute(image)
    sitk.WriteImage(image, out_dir + '/combined.nii')

predict('../data/in/AGALLIJ/scan', 'C:\\Users\\ghesio\\Desktop\\out')
