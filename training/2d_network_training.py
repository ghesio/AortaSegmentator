import os
# see https://github.com/qubvel/segmentation_models/issues/374
os.environ['SM_FRAMEWORK'] = 'tf.keras'
# https://github.com/qubvel/segmentation_models
import segmentation_models as sm
from keras.layers import Input, Conv2D
from utils import data_loader as dl
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

# define backbone for the net
backbone = 'resnet34'
# define which network to train
direction = "axial"
# network parameter
batch_size = 16
epochs = 2
# load data
(x_train, x_test, x_val, y_train, y_test, y_val) = dl.get_data_set(direction=direction, augmentation=False)
# add channel info to train set
x_train = tf.expand_dims(x_train, axis=-1)
# load model
base_model = sm.Unet(backbone, encoder_weights='imagenet')
# define number of channels
N = x_train.shape[-1]
inp = Input(shape=(None, None, N))
l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
out = base_model(l1)
model = Model(inp, out, name=base_model.name)
# compile the model
model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])
# define callbacks for saving checkpoints
callbacks = [
    # this callback saves a SavedModel every X batches
    ModelCheckpoint(
        filepath='../checkpoints/'+direction+'_'+backbone+'-loss={loss:.2f}',
        save_freq=1)
]
# fit the model
history = model.fit(
   x=x_train,
   y=y_train,
   batch_size=batch_size,
   epochs=epochs,
   validation_data=(x_val, y_val),
   callbacks=callbacks
)

print(history.history)
results = model.evaluate(x_test, y_test, batch_size=16)
print(results)
print("test loss, test acc:", results)

