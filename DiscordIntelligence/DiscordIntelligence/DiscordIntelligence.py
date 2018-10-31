from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
import os
import asyncio
import sys
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
import argparse

import Trainer
import ModelHandler

ap = argparse.ArgumentParser();
ap.add_argument('--data-dir', default='res/study-data.txt')
ap.add_argument('--alpha-dir', default='res/alphabet.txt')
ap.add_argument('--model', default=os.getcwd()+'\\checkpoint.hdf5')
args = vars(ap.parse_args())

DATA_DIR = args['data_dir']
ALPHA_DIR = args['alpha_dir']
MODEL = args['model']

BATCH_SIZE = 50
HIDDEN_DIM = 500
SEQ_LENGTH = 50
LAYER_NUM = 2
EPOCHS = 50

X, y, VOCAB_SIZE, ix_to_char, chars = load_data(DATA_DIR, ALPHA_DIR, SEQ_LENGTH)

def init(retrain):
	willLoadModel = False
	mdh = ModelHandler(MODEL, HIDDEN_DIM, VOCAB_SIZE, LAYER_NUM)
	mdl = None
	if not MODEL == '':
		willLoadModel = os.path.isfile(MODEL)
	if willLoadModel and not retrain:
		mdl = mdh.load()
	else:
		mdl = mdh.create()
