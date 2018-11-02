import threading
import os
import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
import ModelHandler
from Utils import *

class Trainer(threading.Thread):

	def __init__(self, modelPath, model, DATA_DIR, ALPHABET_DIR, SEQ_LENGTH = 50, batch_size = 50, epochs = 50):
		self.modelPath = modelPath
		self.model = model
		self.batch_size = batch_size
		self.epochs = epochs
		X, y, VOCAB_SIZE, ix_to_char, chars = load_data(DATA_DIR, ALPHABET_DIR, SEQ_LENGTH)
		self.X = X
		self.y = y
		self.vocsize = VOCAB_SIZE
		self.ix_to_char = ix_to_char
		self.chars = chars

	def retrain(self):
		try:
			os.remove(modelPath)
		except Exception:
			pass
		train()
	
	def train(self):
		self.model.fit(self.X, self.y, batch_size=self.batch_size, verbose=1, epochs=self.epochs)
		self.model.save(self.modelPath, overwrite=True)

	def generate(self, initx, length):
		if initx == None:
			ix = [np.random.randint(self.vocsize)]
		else:
			ix = [initx]
		y_char = [self.ix_to_char[ix[-1]]]
		X = np.zeros((1, length, self.vocsize))
		for i in range(length):
			X[0, i, :][ix[-1]] = 1
			ix = np.argmax(self.model.predict(X[:, :i+1, :])[0], 1)
			y_char.append(self.ix_to_char[ix[-1]])
		return ('').join(y_char)