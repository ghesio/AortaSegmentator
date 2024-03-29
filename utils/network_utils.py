# https://segmentation-models.readthedocs.io/en/latest/tutorial.html#training-with-non-rgb-data
import os

# see https://github.com/qubvel/segmentation_models/issues/374
os.environ['SM_FRAMEWORK'] = 'tf.keras'
# https://github.com/qubvel/segmentation_models
import segmentation_models as sm
from keras.layers import Input, Conv2D
from keras.models import Model, load_model
from keras.optimizers import Adam

# define backbone for the networks
backbone = 'resnet101'
architecture = 'linknet'  # unet, pspnet, linknet, fpn
checkpoint_dir = 'checkpoints/'


def get_preprocessor():
    return sm.get_preprocessing(backbone)


def get_model(number_of_channel=1):
    # load model
    if architecture == 'fpn':
        base_model = sm.FPN(backbone, encoder_weights='imagenet')
    elif architecture == 'pspnet':
        base_model = sm.PSPNet(backbone, encoder_weights='imagenet')
    elif architecture == 'linknet':
        base_model = sm.Linknet(backbone, encoder_weights='imagenet')
    else:
        base_model = sm.Unet(backbone, encoder_weights='imagenet')
    inp = Input(shape=(None, None, number_of_channel))
    l1 = Conv2D(3, (1, 1))(inp)  # map N channels data to 3 channels
    out = base_model(l1)
    model = Model(inp, out, name=base_model.name)
    # print model summary
    model.summary()
    # compile the model
    opt = Adam(learning_rate=0.00001)
    model.compile(optimizer=opt, loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])
    return model


def get_best_checkpoints():
    # read all checkpoint files
    checkpoint_files = [file for file in os.listdir(checkpoint_dir + '/' + architecture) if backbone in file]
    # score list
    scores = []
    for file in checkpoint_files:
        score_value = file[-9:-5]
        if 'axial' in file:
            scores.append(('axial', float(score_value)))
        if 'coronal' in file:
            scores.append(('coronal', float(score_value)))
        if 'sagittal' in file:
            scores.append(('sagittal', float(score_value)))
    # calculate best loss scores
    axial_min_score = min((score for score in scores if score[0] == 'axial'), key=lambda scores: scores[1])[1]
    coronal_min_score = min((score for score in scores if score[0] == 'coronal'), key=lambda scores: scores[1])[1]
    sagittal_min_score = min((score for score in scores if score[0] == 'sagittal'), key=lambda scores: scores[1])[1]
    best_files = []
    for file in checkpoint_files:
        if 'axial' in file and str(axial_min_score) in file:
            best_files.append(checkpoint_dir + '/' + architecture + '/' + file)
        if 'coronal' in file and str(coronal_min_score) in file:
            best_files.append(checkpoint_dir + '/' + architecture + '/' + file)
        if 'sagittal' in file and str(sagittal_min_score) in file:
            best_files.append(checkpoint_dir + '/' + architecture + '/' + file)
    return best_files


def get_pretrained_models():
    models = []
    best_files = get_best_checkpoints()
    for i in range(len(best_files)):
        models.append(load_model(best_files[i], compile=False))
    return models


if __name__ == "__main__":
    get_model(1)
    exit(0)
