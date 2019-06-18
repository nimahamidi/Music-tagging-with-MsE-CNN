'''
MsE-CNN for Music Tagging in Keras
Nima Hamidi - April 2019
'''

from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, BatchNormalization, ELU
from keras.layers.merge import Concatenate

'''-------------------
   Required functions
---------------------'''
def concat (L1, L2):
    L = Concatenate()([L1, L2])
    return L

'''-------------------------------------------
Traditional CNN for Music tagging
This model has been developped by Keunwoo Choi
This model is baseline for MsE-CNN model
--------------------------------------------'''
def MusicTaggerCNN(weights='msd', input_tensor=None, include_top=True):

    input_shape = (96, 1366, 1)
    melgram_input = Input(shape=input_shape)
    channel_axis = 3
    freq_axis = 1
    time_axis = 2

    # Input block
    x = BatchNormalization(name='bn_0_freq', axis=1)(melgram_input)

    # Conv block 1
    x = Conv2D(64, (3, 3), padding='same', name='conv1')(x)
    x = BatchNormalization(name='bn1', axis=3)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool1')(x)

    # Conv block 2
    x = Conv2D(128, (3, 3), padding='same', name='conv2')(x)
    x = BatchNormalization(name='bn2', axis=3)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool2')(x)

    # Conv block 3
    x = Conv2D(128, (3, 3), padding='same', name='conv3')(x)
    x = BatchNormalization(axis=channel_axis, name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool3')(x)

    # Conv block 4
    x = Conv2D(128, (3, 3), padding='same', name='conv4')(x)
    x = BatchNormalization(axis=channel_axis, name='bn4')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 5), name='pool4')(x)

    # Conv block 5
    x = Conv2D(64, (3, 3), padding='same', name='conv5')(x)
    x = BatchNormalization(name='bn5', axis=3)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), name='pool5')(x)

    # Output
    x = Flatten()(x)
    if include_top:
        x = Dense(50, activation='sigmoid', name='output')(x)

    # Create model
    model = Model(melgram_input, x)
    if weights is None:
        return model
    else:
        model.load_weights('data/cnn_weights.h5',by_name=True)
        return model

'''-------------------------------------------
Multi-scale CNN for Music tagging
This model has been developped by Nima Hamidi
This is a keras base model for music Tagging based on a the paper
"Multi-scale CNN for Music Tagging, accepted at ML4MD at ICML"
--------------------------------------------'''
def MS_CNN_MusicTagger(weights='msd', input_tensor=None, include_top=True):

    input_shape = (96, 1366, 1)
    melgram_input = Input(shape=input_shape)
    channel_axis = 3
    freq_axis = 1
    time_axis = 2

    # Input block
    x = BatchNormalization(name='bn_0_freq', axis=1)(melgram_input)
    x_g = MaxPooling2D(pool_size=(2, 4), name='pool1_g')(x)

    # Conv block 1
    x = Conv2D(64, (3, 3), padding='same', name='conv1')(x)
    x = BatchNormalization(name='bn1', axis=3)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool1')(x)
    x_comb = concat(x_g, x)
    x_g = MaxPooling2D(pool_size=(2, 4), name='pool1_comb')(x_comb)

    # Conv block 2
    x = Conv2D(128, (3, 3), padding='same', name='conv2')(x)
    x = BatchNormalization(name='bn2', axis=3)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool2')(x)
    x_comb = concat(x_g, x)
    x_g = MaxPooling2D(pool_size=(2, 4), name='pool2_comb')(x_comb)

    # Conv block 3
    x = Conv2D(256, (3, 3), padding='same', name='conv3')(x)
    x = BatchNormalization(name='bn3', axis=3)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool3')(x)
    x_comb = concat(x_g, x)
    x_g = MaxPooling2D(pool_size=(3, 5), name='pool3_comb')(x_comb)
    # Conv block 4
    x = Conv2D(512, (3, 3), padding='same', name='conv4')(x)
    x = BatchNormalization(name='bn4', axis=3)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 5), name='pool4')(x)
    x_comb = concat(x_g, x)
    x_g = MaxPooling2D(pool_size=(4, 4), name='pool4_comb')(x_comb)

    # Conv block 5
    x = Conv2D(1024, (3, 3), padding='same', name='conv5')(x)
    x = BatchNormalization(name='bn5', axis=3)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), name='pool5')(x)
    x = concat(x_g, x)

    # Output
    x = Flatten()(x)
    if include_top:
        x = Dense(50, activation='sigmoid', name='output')(x)

    # Create model
    model = Model(melgram_input, x)

    if weights is None:
        return model
    else:
        model.load_weights('data/MScnn_weights.h5',by_name=True)
        return model

if __name__ == "__main__":
    model = MS_CNN_MusicTagger(weights=None, input_tensor=None, include_top=True)
