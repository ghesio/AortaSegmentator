import os

# see https://github.com/qubvel/segmentation_models/issues/374
os.environ['SM_FRAMEWORK'] = 'tf.keras'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# https://github.com/qubvel/segmentation_models
import tensorflow as tf
from datetime import datetime
from utils.network_utils import get_model, backbone
from keras.preprocessing.image import ImageDataGenerator
import imageio
import numpy as np

# dry run flag
dry_run = False
separator = "/"
# define which network to train
directions = ['axial']  # , 'coronal', 'sagittal']
# network parameter
batch_size = 50
epochs = 20


def zip_generator(image_data_generator, mask_data_generator):
    zipped_generator = zip(image_data_generator, mask_data_generator)
    for (img, mask) in zipped_generator:
        yield img[0] / 255, mask[0] / 255


for direction in directions:
    print('Start training for direction ' + direction + ' @ ' + datetime.now().strftime("%H:%M:%S"))
    train_scans_dir = 'data/slices/train/' + direction + separator
    train_labels_dir = 'data/slices/train/' + direction + separator
    validation_scans_dir = 'data/slices/validation/' + direction
    validation_labels_dir = 'data/slices/validation/' + direction
    test_scans_dir = 'data/slices/test/' + direction
    test_labels_dir = 'data/slices/test/' + direction
    data_shape = np.array(imageio.imread(uri=train_scans_dir + 'scans' + separator + direction + '_00000000.png'),
                          dtype='uint8').shape

    number_of_train_samples = len(os.listdir(train_scans_dir + 'scans'))
    print('Number of slices in train ' + str(number_of_train_samples))
    number_of_validation_samples = len(os.listdir(validation_scans_dir + 'scans'))
    print('Number of slices in validation ' + str(number_of_validation_samples))
    number_of_test_samples = len(os.listdir(validation_labels_dir + 'scans'))
    print('Number of slices in test ' + str(number_of_test_samples))


    # instantiate data generators
    data_gen_args = dict(
        rotation_range=5,
        width_shift_range=5,
        height_shift_range=5,
        zoom_range=[0.8, 1.3],
    )
    # TRAIN SET
    train_scan_generator = ImageDataGenerator(data_gen_args).flow_from_directory(train_scans_dir,
                                                                                 batch_size=batch_size,
                                                                                 seed=42,
                                                                                 color_mode='grayscale',
                                                                                 target_size=data_shape,
                                                                                 classes=['scans'])
    train_mask_generator = ImageDataGenerator(data_gen_args).flow_from_directory(train_labels_dir,
                                                                                 batch_size=batch_size,
                                                                                 seed=42,
                                                                                 color_mode='grayscale',
                                                                                 target_size=data_shape,
                                                                                 classes=['labels'])
    train_generator = zip_generator(train_scan_generator, train_mask_generator)
    # VALIDATION
    validation_scan_generator = ImageDataGenerator().flow_from_directory(
        validation_scans_dir,
        shuffle=False,
        batch_size=batch_size,
        color_mode='grayscale',
        target_size=data_shape,
        classes=['scans'])
    validation_label_generator = ImageDataGenerator().flow_from_directory(
        validation_labels_dir,
        shuffle=False,
        batch_size=batch_size,
        color_mode='grayscale',
        target_size=data_shape,
        classes=['labels'])
    validation_generator = zip_generator(validation_scan_generator, validation_label_generator)
    # TEST
    test_scan_generator = ImageDataGenerator().flow_from_directory(test_scans_dir,
                                                                   shuffle=False,
                                                                   batch_size=batch_size,
                                                                   color_mode='grayscale',
                                                                   target_size=data_shape,
                                                                   classes=['scans'])
    test_label_generator = ImageDataGenerator().flow_from_directory(test_labels_dir,
                                                                    shuffle=False,
                                                                    batch_size=batch_size,
                                                                    color_mode='grayscale',
                                                                    target_size=data_shape,
                                                                    classes=[
                                                                        'labels'])
    test_generator = zip_generator(test_scan_generator, test_label_generator)
    if dry_run:
        continue
    # get model
    model = get_model()
    # define callbacks
    # a) save checkpoints
    save_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='checkpoints/' + direction + '_'
                 + backbone + '-{epoch:02d}-{val_loss:.2f}.hdf5',
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    # b) early stopping criteria
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3)
    # fit the model
    print('Training start @ ', datetime.now().strftime("%H:%M:%S"), ' - ', direction)
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        steps_per_epoch=number_of_train_samples // batch_size,
        validation_steps=number_of_validation_samples // batch_size,
        callbacks=[save_callback, early_stopping_callback]
    )
    print("Training end @ ", datetime.now().strftime("%H:%M:%S"))
    print("\r\nTraining results")
    for key in history.history.keys():
        print('\t' + key + ':', history.history[key])
    print("\r\nTest evaluation")
    results = model.evaluate(test_generator, batch_size=batch_size)
    print('\tLoss:', results[0], 'Accuracy:', results[1])
exit(0)
