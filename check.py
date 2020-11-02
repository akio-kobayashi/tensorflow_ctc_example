import os
import numpy as np
import argparse
#import kaldi_io_py
import subprocess
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True, help='input file')
args = parser.parse_args()

with h5py.File(args.input,'r') as f:
    keys = f.keys()
    sum=[]
    sum2=[]
    num=0
    for key in keys:
        data=f[key+'/label'][()]
        print(key)
        print(data)
