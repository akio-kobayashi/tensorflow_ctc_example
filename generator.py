import numpy as np
import sys, os, re, gzip, struct
import random
import h5py
import copy
import keras.backend as K
from keras.utils import Sequence
import keras.utils
from keras.preprocessing import sequence
import tensorflow as tf

class DataGenerator(Sequence):

    def __init__(self, file, batch_size=64, feat_dim=40, n_labels=1024, shuffle=True):

        self.file=file
        self.batch_size=batch_size
        self.feat_dim=feat_dim
        self.n_labels=n_labels
        self.shuffle=shuffle
        self.keys=[]
        
        self.h5fd = h5py.File(self.file, 'r')
        self.n_samples = len(self.h5fd.keys())
        for key in self.h5fd.keys():
            self.keys.append(key)

        if self.shuffle:
            random.shuffle(self.keys)
            
    def __len__(self):
        # for simple implementation
        return int(np.ceil(self.n_samples)/self.batch_size)


    def __getitem__(self, index, return_keys=False):
        list_keys_temp = [self.keys[k] for k in range(index*self.batch_size,
                                                      min( (index+1)*self.batch_size,
                                                      len(self.keys) ) )]

        # [input_sequences, label_sequences, inputs_lengths, labels_length]
        input_sequences, label_sequences, inputs_lengths, labels_lengths = self.__data_generation(list_keys_temp)

        if return_keys is False:
            return [input_sequences, label_sequences, inputs_lengths, labels_lengths]
        else:
            return [input_sequences, label_sequences, inputs_lengths, labels_lengths], list_keys_temp

    def on_epoch_end(self):
        if self.shuffle == True:
            # shuffling start pointers
            random.shuffle(self.keys)

    def __data_generation(self, list_keys_temp):

        max_input_len=0
        max_output_len=0
        labels=[]
        in_seq=[]
        lb_seq=[]

        for i, key in enumerate(list_keys_temp):
            mat = self.h5fd[key+'/data'][()]
            if mat.shape[0] > max_input_len:
              max_input_len = mat.shape[0]
            in_seq.append(mat.shape[0])
            
            # label is a list of integers starting from 0
            label = self.h5fd[key+'/label'][()]
            #label += 1 # for starting blank from 0
            labels.append(np.array(label))
            if len(label) > max_output_len:
              max_output_len = len(label)
            lb_seq.append(len(label))

        input_sequences = np.zeros((self.batch_size, max_input_len, self.feat_dim))
        for i, key in enumerate(list_keys_temp):
            mat = self.h5fd[key+'/data'][()]
            input_sequences[i, 0:mat.shape[0], :] = np.expand_dims(mat, axis=0)

        label_sequences=sequence.pad_sequences(labels, maxlen=max_output_len,
                                               padding='post', value=0)
        inputs_lengths=np.array(in_seq)
        labels_lengths=np.array(lb_seq)

        return input_sequences, label_sequences, inputs_lengths, labels_lengths
