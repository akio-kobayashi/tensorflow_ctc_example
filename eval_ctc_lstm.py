import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.simplefilter('ignore')
import argparse
import sys
import subprocess
import time
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
import keras.utils
import keras.backend as K
import numpy as np
import random
import generator
import vgg2l
import network
import Levenshtein

os.environ['PYTHONHASHSEED']='0'
np.random.seed(1024)
random.seed(1024)
np.set_printoptions(threshold=np.inf)

config = tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(allow_growth = True),
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)

max_label_len=1024

def id2char(ids, char_map):
    tokens=[]
    for id in ids:
        if id < len(char_map):
            tokens.append(char_map[int(id)])
    return tokens

def read_map(file, char_map):
    with open(file, "r") as f:
        lines = f.read().splitlines()
        for i, line in enumerate(lines):
            char_map[i] = line.strip()

def named_logs(model, logs):
  result = {}
  for l in zip(model.metrics_names, logs):
    result[l[0]] = l[1]
  return result

def cer(ref, tgt):
    return Levenshtein.distance(ref, tgt)/len(ref) * 100.0

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='data')
    parser.add_argument('--feat-dim', default=40, type=int, help='feats dim')
    parser.add_argument('--n-labels', default=1024, type=int, required=True,
                        help='number of output labels')
    parser.add_argument('--batch-size', default=16, type=int, help='mini-batch size')
    #parser.add_argument('--snapshot', type=str, default='./',
    #                    help='snapshot directory')
    #parser.add_argument('--snapshot-prefix', type=str, default='eval',
    #                    help='snapshot file prefix')
    parser.add_argument('--result', type=str, default='result.txt',help='results')
    parser.add_argument('--weights', type=str, required=True, help='model weights')
    parser.add_argument('--units', type=int ,default=16, help='number of LSTM cells')
    parser.add_argument('--lstm-depth', type=int ,default=2,
                        help='number of LSTM/GRU layers')
    parser.add_argument('--filters', type=int, default=64, \
                        help='number of filters for CNNs')
    parser.add_argument('--char', type=str,
                        help='character map')
    
    args = parser.parse_args()

    inputs = Input(shape=(None, args.feat_dim))
    model = network.build_model(inputs, args.units, args.lstm_depth, args.n_labels,
                                args.feat_dim, 0.0, 0.0, args.filters, None)
    model.load_weights(args.weights, by_name=True)

    test_generator = generator.DataGenerator(args.data,
                                             args.batch_size,
                                             args.feat_dim, args.n_labels, False)

    char_map={}
    read_map(args.char, char_map)

    with open(args.result, 'w') as f:
        for bt in range(test_generator.__len__()):
            data, keys = test_generator.__getitem__(bt, return_keys=True)
            predict = model.predict_on_batch(data[0])
            argmax=tf.keras.backend.eval(tf.keras.backend.argmax(predict, axis=2))
        
            for i in range(argmax.shape[0]):
                decode=[]
                correct=[]
            
                for k in range(argmax.shape[1]):
                    index = argmax[i,k]
                    if index != args.n_labels+1:
                        if k != 0 and index == argmax[i,k-1]:
                            continue
                        decode.append(argmax[i,k])
                    
                hyp=id2char(decode, char_map)
                hyp="".join(hyp)
                label=data[1]
                for k in range(label.shape[1]):
                    correct.append(label[i,k])
                ref=id2char(correct, char_map)
                ref="".join(ref)
                score=cer(ref, hyp)
                line = "Ref:   %s\n" % ref
                f.write(line)
                line = "Hyp:   %s\n" % hyp
                f.write(line)
                line = "CER:   %.3f\n\n" % score
                f.write(line)
                
if __name__ == "__main__":
    main()
