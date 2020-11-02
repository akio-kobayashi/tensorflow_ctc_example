import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.simplefilter('ignore')
import numpy as np
import argparse
import subprocess
import h5py
import tensorflow as tf
import keras

print(tf.__version__)
print(keras.__version__)
