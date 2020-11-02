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
import time
import CTCModel
import generator
import vgg2l
import network

os.environ['PYTHONHASHSEED']='0'
np.random.seed(1024)
random.seed(1024)

config = tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(allow_growth = True),
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)

def write_log(callback, names, logs, batch_no):
    for name, value in zip (names, logs):
        summary = tf.compat.v1.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()
        
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='training data')
    parser.add_argument('--valid', type=str, required=True, help='validation data')
    parser.add_argument('--feat-dim', default=40, type=int, help='feats dim')
    parser.add_argument('--n-labels', default=1024, type=int, required=True,
                        help='number of output labels')
    parser.add_argument('--batch-size', default=64, type=int, help='mini-batch size')
    parser.add_argument('--epochs', default=10, type=int, help='training epochs')
    parser.add_argument('--snapshot', type=str, default='./',
                        help='snapshot directory')
    parser.add_argument('--snapshot-prefix', type=str, default='snapshot',
                        help='snapshot file prefix')
    parser.add_argument('--learn-rate', type=float, default=1.0e-3,
                        help='initial learn rate')
    #parser.add_argument('--log-dir', type=str, default='./',
    #                    help='tensorboard log directory')
    parser.add_argument('--units', type=int ,default=16, help='number of LSTM cells')
    parser.add_argument('--lstm-depth', type=int ,default=2,
                        help='number of LSTM layers')
    parser.add_argument('--factor', type=float, default=0.5,help='lerarning rate decaying factor')
    parser.add_argument('--min-lr', type=float, default=1.0e-6, help='minimum learning rate')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--filters', type=int, default=64, help='number of filters for CNNs')
    parser.add_argument('--max-patience', type=int, default=5, help='max patient')
    parser.add_argument('--optim', type=str, default='adam', help='optimizer [adam|adadelta]')
    parser.add_argument('--weights', type=str, default=None, help='network werigts')
    args = parser.parse_args()

    inputs = Input(shape=(None, args.feat_dim))
    curr_lr = args.learn_rate
    model = network.build_model(inputs, args.units, args.lstm_depth, args.n_labels,
                                args.feat_dim, curr_lr, args.dropout, args.filters, args.optim)

    if args.weights is not None:
        model.load_weights(args.weights, by_name=True)

    tensorboard = keras.callbacks.TensorBoard(
        log_dir='./tensorboard',
        histogram_freq=0,
        batch_size=args.batch_size,
        write_graph=True,
        write_grads=True
    )
    tensorboard.set_model(model)

    training_generator = generator.DataGenerator(args.data, args.batch_size, args.feat_dim,
                                                 args.n_labels, shuffle=True)
    valid_generator = generator.DataGenerator(args.valid, args.batch_size, args.feat_dim,
                                              args.n_labels, shuffle=False)
    ep=0
   
    train_bt=0
    val_bt=0
    while ep < args.epochs:
        start_time=time.time()
        curr_loss = 0.0
        curr_samples=0
        
        for bt in range(training_generator.__len__()):
            #print("batch %d/%d" % (bt+1, training_generator.__len__()))
            data = training_generator.__getitem__(bt)
            # data = [input_sequences, label_sequences, inputs_lengths, labels_length]
            # y (true labels) is set to None, because not used in tensorflow CTC training.
            # 'train_on_batch' will return CTC-loss value
            logs = model.train_on_batch(x=data,y=data[1])
            _logs=[]
            _logs.append(logs)
            train_bt+=bt
            write_log(tensorboard, ['loss'] , _logs, train_bt)
            
        curr_val_cer = []
        for bt in range(valid_generator.__len__()):
            data = valid_generator.__getitem__(bt)
            # eval_on_batch will return sequence error rate (ser) and label error rate (ler)
            # the function returns ['loss', 'ler', 'ser']
            # 'ler' should not be normalized by true lengths
            _logs=[]
            loss, cer, ser = model.evaluate(data)
            _logs.append(loss[0])
            _logs.append(np.mean(np.array(cer)))
            _logs.append(np.mean(np.array(ser)))
            val_bt+=bt
            write_log(tensorboard, ['val_loss', 'val_cer', 'val_ser'], _logs, val_bt)
            curr_val_cer.append(cer)
            
        curr_val_cer = np.mean(curr_val_cer)*100.0
        print('Epoch %d (valid) cer=%.4f' % (ep+1, curr_val_cer))
            
        path = os.path.join(args.snapshot,args.snapshot_prefix+'.h5')
        model.save_weights(path)
        msg="save the model epoch %d" % (ep+1)
        ep += 1
            

    print("Training End.")

if __name__ == "__main__":
    main()
