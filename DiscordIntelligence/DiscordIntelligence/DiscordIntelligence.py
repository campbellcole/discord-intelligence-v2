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

sys.path.append('./TextRNN')

from Trainer import Trainer
from ModelHandler import ModelHandler
from Utils import *

ap = argparse.ArgumentParser();
ap.add_argument('--data-dir', default='res\\study-data.txt', help='file to read study data from')
ap.add_argument('--alpha-dir', default='res\\alphabet.txt', help='file containing alphabet to use')
ap.add_argument('--model', default='res\\network.hdf5', help='path to model to load')
ap.add_argument('--debug', action='store_true', help='debug/verbose mode')
args = vars(ap.parse_args())

DATA_DIR = args['data_dir']
ALPHA_DIR = args['alpha_dir']
MODEL = args['model']
DEBUG = args['debug']

BATCH_SIZE = 50
HIDDEN_DIM = 500
SEQ_LENGTH = 50
LAYER_NUM = 2
EPOCHS = 50

def debug(out):
	if DEBUG:
		print(out)

debug('Loading model...')
mdh = ModelHandler(MODEL)
mdl = None
if os.path.isfile(MODEL):
	debug('Found model. Loading...')
	mdl = mdh.load()
else:
	debug("Couldn't find model. Creating...")
	mdl = mdh.create(HIDDEN_DIM, VOCAB_SIZE, LAYER_NUM)

debug('Creating trainer...')
trainer = Trainer(MODEL, mdl, DATA_DIR, ALPHA_DIR)

s = ""
while not s == "exit":
	s = input('--> ')
	cmd = s.split(" ")
	if cmd[0] == 'gen':
		genlen = 10
		initx = None
		try:
			genlen = int(cmd[1])
		except Exception:
			pass
		try:
			if cmd[2] in chars:
				initx = chars.cmd(cmd[2])
		except Exception:
			pass
		print("Generating...")
		print(trainer.generate(initx, genlen))
	if cmd[0] == 'train':
		try:
			EPOCHS = int(cmd[1])
		except Exception:
			pass
		print('Training {} times...'.format(EPOCHS))
		trainer.epochs = EPOCHS
		trainer.train()

debug('Exiting...')