# This code is imported from the following project: https://github.com/asmith26/wide_resnets_keras

import logging
import sys
import numpy as np
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Activation, add, Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from mobilenet_sipeed.mobilenet import MobileNet

sys.setrecursionlimit(2 ** 20)
np.random.seed(2 ** 10)


class _MobileNet:
    def __init__(self, depth=16):
        self._depth = depth
        self._dropout_probability = 0
        self._weight_decay = 0.0005
        self._use_bias = False
        self._weight_init = "he_normal"


    def __call__(self):
    
        logging.debug("Creating model...")
        input_image = Input(shape=(128, 128, 3))
        mobilenet = MobileNet(input_shape=(128,128,3), input_tensor=input_image, alpha = 0.75, weights = None, include_top=False, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
        # Classifier block
        x=GlobalAveragePooling2D()(mobilenet.outputs[0])
        #x = Flatten()(mobilenet.outputs[0])
        fc_g = x #Dense(128, activation="relu")(x)
        fc_a = x #Dense(32, activation="relu")(x)
        predictions_g = Dense(2, activation="softmax", name="pred_gender")(fc_g)
        predictions_a = Dense(5, activation="softmax", name="pred_age")(fc_a)
        model = Model(inputs=input_image, outputs=[predictions_g, predictions_a])
        #model = Model(inputs=input_image, outputs=predictions_g)

        return model

def main():
    model = MobileNet(64)()
    model.summary()


if __name__ == '__main__':
    main()
