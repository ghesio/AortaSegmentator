# https://segmentation-models.readthedocs.io/en/latest/tutorial.html#training-with-non-rgb-data
import os
# see https://github.com/qubvel/segmentation_models/issues/374
os.environ['SM_FRAMEWORK'] = 'tf.keras'
# https://github.com/qubvel/segmentation_models
import segmentation_models as sm
from keras.layers import Input, Conv2D
from keras.models import Model


def get_model(number_of_channel=1, backbone='resnet34'):
    # load model
    base_model = sm.Unet(backbone, classes=2, encoder_weights='imagenet')
    inp = Input(shape=(None, None, number_of_channel))
    l1 = Conv2D(3, (1, 1))(inp)  # map N channels data to 3 channels
    out = base_model(l1)
    model = Model(inp, out, name=base_model.name)
    # print model summary
    model.summary()
    # compile the model
    model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])
    return model


if __name__ == "__main__":
    get_model()
    exit(0)