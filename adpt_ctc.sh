#!/bin/sh

#export CUDA_VISIBLE_DEVICES=$device

train=tensorflow_ctc_example_data/valid_50.h5
valid=tensorflow_ctc_example_data/eval_10.h5

n_labels=82

# features
feat_dim=40
units=160
lstm_depth=3

#training
batch_size=10
epochs=5
learn_rate=1.0e-5
weights=tensorflow_ctc_example_data/model/snapshot.h5

snapdir=./out/
mkdir -p $snapdir
mkdir -p tensorboard

python ctc_lstm.py --data $train --valid $valid \
       --feat-dim $feat_dim --n-labels $n_labels \
       --batch-size $batch_size --epochs $epochs \
       --snapshot $snapdir  --learn-rate $learn_rate \
       --units $units --lstm-depth $lstm_depth --weights $weights
