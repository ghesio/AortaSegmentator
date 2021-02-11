import os
# see https://github.com/qubvel/segmentation_models/issues/374
os.environ['SM_FRAMEWORK'] = 'tf.keras'
# https://github.com/qubvel/segmentation_models
import segmentation_models as sm
from keras.layers import Input, Conv2D
import data_preprocessing.data_loader as dl
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from datetime import datetime
from utils.network_utils import get_model, backbone

# define which network to train
direction = "sagittal"
# network parameter
batch_size = 16
epochs = 1
# data parameter
samples_from_each_patient = 20
# load data
(x_train, y_train) = dl.get_train_set(direction=direction, samples_from_each_patient=samples_from_each_patient, augmentation=False)
(x_val, y_val) = dl.get_validation_set(direction=direction, samples_from_each_patient=samples_from_each_patient)

# add channel info to train set
x_train = tf.expand_dims(x_train, axis=-1)
# get model
model = get_model(x_train.shape[-1])
# define callbacks
# a) save checkpoints
save_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='../checkpoints/'+direction+'_'+str(samples_from_each_patient)+'_'+backbone+'-{epoch:02d}-{val_loss:.2f}.hdf5',
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
# b) early stopping criteria
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3)
# fit the model
print('Training start @', datetime.now().strftime("%H:%M:%S"), '-', direction)
history = model.fit(
   x=x_train,
   y=y_train,
   batch_size=batch_size,
   epochs=epochs,
   validation_data=(x_val, y_val),
   callbacks=[save_callback, early_stopping_callback]
)
print("Training end @ ", datetime.now().strftime("%H:%M:%S"))
(x_test, y_test) = dl.get_validation_set(direction=direction, samples_from_each_patient=samples_from_each_patient)
print("\r\nTraining results")
for key in history.history.keys():
    print('\t' + key+':', history.history[key])
print("\r\nTest evaluation")
results = model.evaluate(x_test, y_test, batch_size=16)
print('\tLoss:', results[0], 'Accuracy:', results[1])
exit(0)
