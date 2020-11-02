#!/bin/sh

#export CUDA_VISIBLE_DEVICES=$device

data=eval_10.h5

# number of labels w/o blank
n_labels=82

# features
feat_dim=40
units=160
lstm_depth=3

#training
batch_size=1
learn_rate=1.0e-5;
snapdir=./out
model=$snapdir/snapshot.h5
python eval_ctc_lstm.py --data $data \
       --feat-dim $feat_dim --n-labels $n_labels --batch-size $batch_size \
       --units $units --lstm-depth $lstm_depth --weights $model \
       --char char.txt
