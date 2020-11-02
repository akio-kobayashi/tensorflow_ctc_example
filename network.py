from keras.models import Model
from keras.layers import Dense,Input,LSTM,CuDNNLSTM,Activation,CuDNNGRU,GRU
from keras.layers import TimeDistributed, Bidirectional
import keras.utils
import keras.backend as K
import numpy as np
import tensorflow as tf
import CTCModel
import vgg2l

cudnn=False

def lstm(outputs, units, depth, n_labels, dropout):
    for n in range (depth):
        if lstm is True:
            if cudnn is True:
                outputs=Bidirectional(CuDNNLSTM(units,
                                                return_sequences=True))(outputs)
            else:
                outouts=Bidirectional(LSTM(units, kernel_initializer='glorot_uniform',
                                           return_sequences=True,
                                           use_forget_bias=True,
                                           dropout=dropout,
                                           unroll=False))(outputs)
        else:
            if cudnn is False:
                outputs=Bidirectional(GRU(units,
                                          kernel_initializer='glorot_uniform',
                                          return_sequences=True,
                                          dropout=dropout,
                                          unroll=False))(outputs)
            else:
                outputs=Bidirectional(CuDNNGRU(units,
                                               return_sequences=True))(outputs)
    return outputs

def build_model(inputs, units, depth, n_labels, feat_dim, init_lr,
                dropout, init_filters, optim):

    outputs = vgg2l.VGG2L(inputs, init_filters, feat_dim)

    outputs = lstm(outputs,units, depth, n_labels, dropout)
    outputs = TimeDistributed(Dense(n_labels+1))(outputs)
    outputs = Activation('softmax')(outputs)

    if optim is not None:
        model=CTCModel.CTCModel([inputs], [outputs], greedy=True)
        if optim == 'adam':
            model.compile(keras.optimizers.Adam(lr=init_lr))
        elif optim == 'sgd':
            model.compile(keras.optimizers.SGD(lr=init_lr,  momentum=0.9))
        else:
            model.compile(keras.optimizers.Adadelta(lr=init_lr))
    else:
        model=Model(inputs, outputs)
    return model
