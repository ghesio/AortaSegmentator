import os

# see https://github.com/qubvel/segmentation_models/issues/374
os.environ['SM_FRAMEWORK'] = 'tf.keras'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# https://github.com/qubvel/segmentation_models
import tensorflow as tf
from datetime import datetime
from utils.network_utils import get_model, backbone, architecture, get_preprocessor
from keras.preprocessing.image import ImageDataGenerator
import imageio
import numpy as np

# dry run flag
dry_run = False
separator = "/"
data_slices_root = "data/slices/"
# define which network to train
directions = ['axial', 'coronal', 'sagittal']
# network parameter
batch_sizes = [32, 20, 15]
max_epochs = 60
preprocessor = get_preprocessor()


def zip_generator(image_data_generator, mask_data_generator):
    zipped_generator = zip(image_data_generator, mask_data_generator)
    for (img, mask) in zipped_generator:
        yield np.stack((preprocessor(img[0]),)*3, axis=-1), np.stack((mask[0]/255,)*3, axis=-1)


for i in range(len(directions)):
    direction = directions[i]
    batch_size = batch_sizes[i]
    train_scan_root = data_slices_root + 'train/' + direction + separator
    validation_scan_root = data_slices_root + 'validation/' + direction + separator
    test_scan_root = data_slices_root + 'test/' + direction + separator
    data_shape = np.array(imageio.imread(uri=train_scan_root + 'scans' + separator + direction + '_00000000.png'),
                          dtype='uint8').shape

    number_of_train_samples = len(os.listdir(train_scan_root + 'scans'))
    print('Number of slices in train ' + str(number_of_train_samples))
    number_of_validation_samples = len(os.listdir(validation_scan_root + 'scans'))
    print('Number of slices in validation ' + str(number_of_validation_samples))
    number_of_test_samples = len(os.listdir(test_scan_root + 'scans'))
    print('Number of slices in test ' + str(number_of_test_samples))

    # instantiate data generators
    # TRAIN SET
    train_scan_generator = ImageDataGenerator(dtype='uint8',
                                              rotation_range=5,
                                              width_shift_range=10,
                                              height_shift_range=10,
                                              zoom_range=[0.7, 1.4]
                                              ).flow_from_directory(train_scan_root,
                                                                    batch_size=batch_size,
                                                                    seed=42,
                                                                    color_mode='grayscale',
                                                                    target_size=data_shape,
                                                                    classes=['scans'])
    train_mask_generator = ImageDataGenerator(rotation_range=5,
                                              width_shift_range=10,
                                              height_shift_range=10,
                                              zoom_range=[0.7, 1.4]
                                              ).flow_from_directory(train_scan_root,
                                                                    batch_size=batch_size,
                                                                    seed=42,
                                                                    color_mode='grayscale',
                                                                    target_size=data_shape,
                                                                    classes=['labels'])
    train_generator = zip_generator(train_scan_generator, train_mask_generator)
    # VALIDATION
    validation_scan_generator = ImageDataGenerator(dtype='uint8').flow_from_directory(validation_scan_root,
                                                                                      shuffle=False,
                                                                                      batch_size=batch_size,
                                                                                      color_mode='grayscale',
                                                                                      target_size=data_shape,
                                                                                      classes=['scans'])
    validation_label_generator = ImageDataGenerator().flow_from_directory(validation_scan_root,
                                                                          shuffle=False,
                                                                          batch_size=batch_size,
                                                                          color_mode='grayscale',
                                                                          target_size=data_shape,
                                                                          classes=['labels'])
    validation_generator = zip_generator(validation_scan_generator, validation_label_generator)
    # TEST
    test_scan_generator = ImageDataGenerator(dtype='uint8').flow_from_directory(test_scan_root,
                                                                                shuffle=False,
                                                                                batch_size=batch_size,
                                                                                color_mode='grayscale',
                                                                                target_size=data_shape,
                                                                                classes=['scans'])
    test_label_generator = ImageDataGenerator().flow_from_directory(test_scan_root,
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
        filepath='checkpoints/' + architecture + '/' + direction + '_'
                 + backbone + '-{epoch:02d}-{val_loss:.2f}.hdf5',
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    # b) early stopping criteria
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10,
                                                               restore_best_weights=True)
    # fit the model
    print('Training start @ ', datetime.now().strftime("%H:%M:%S"), ' - ', direction, ' - ', backbone, ' - ',
          architecture, ' - batch size: ', str(batch_size))
    history = model.fit(
        train_generator,
        epochs=max_epochs,
        validation_data=validation_generator,
        steps_per_epoch=number_of_train_samples // batch_size,
        validation_steps=number_of_validation_samples // batch_size,
        callbacks=[save_callback, early_stopping_callback],
        verbose=2
    )
    print("Training end @ ", datetime.now().strftime("%H:%M:%S"))
    print("\r\nTraining results")
    for key in history.history.keys():
        print('\t' + key + ':', history.history[key])
    print("\r\nTest evaluation")
    results = model.evaluate(test_generator, batch_size=batch_size, steps=number_of_test_samples // batch_size,
                             verbose=0)
    print('\tLoss:', results[0], 'Accuracy:', results[1])
exit(0)
