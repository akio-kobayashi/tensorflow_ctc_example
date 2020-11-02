import argparse
import os
import sys
import subprocess
import time
from keras.models import Model
from keras.layers import Dense,Input,BatchNormalization,Softmax,LSTM,Activation, CuDNNGRU, GRU, Reshape
from keras.layers import TimeDistributed, Bidirectional, Dropout, Lambda, Masking, Conv2D, MaxPooling2D
import keras.utils
import keras.backend as K
import numpy as np
import random
import tensorflow as tf
#import ce_generator
#import layer_normalization

def VGG2L(inputs, filters, feat_dim):

    outputs=Lambda(lambda x: tf.expand_dims(x, -1))(inputs)
    outputs=Conv2D(filters=filters,
                   kernel_size=3, padding='same',
                   strides=1,
                   data_format='channels_last',
                   kernel_initializer='glorot_uniform')(outputs)
    outputs=BatchNormalization(axis=-1)(outputs)
    outputs=Activation('relu')(outputs)

    outputs=Conv2D(filters=filters,
                   kernel_size=3, padding='same',
                   strides=1,
                   data_format='channels_last',
                   kernel_initializer='glorot_uniform')(outputs)
    outputs=BatchNormalization(axis=-1)(outputs)
    outputs=Activation('relu')(outputs)

    filters *= 2 # 128
    outputs=Conv2D(filters=filters,
                   kernel_size=3, padding='same',
                   strides=1,
                   data_format='channels_last',
                   kernel_initializer='glorot_uniform')(outputs)
    outputs=BatchNormalization(axis=-1)(outputs)
    outputs=Activation('relu')(outputs)

    outputs=Conv2D(filters=filters,
                   kernel_size=3, padding='same',
                   strides=1,
                   data_format='channels_last',
                   kernel_initializer='glorot_uniform')(outputs)
    outputs=BatchNormalization(axis=-1)(outputs)
    outputs=Activation('relu')(outputs)

    outputs = Reshape(target_shape=(-1, feat_dim*filters))(outputs)

    return outputs
