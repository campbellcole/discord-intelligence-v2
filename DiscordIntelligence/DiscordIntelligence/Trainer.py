import threading
import os
import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
import ModelHandler

class Trainer(threading.Thread):

	def __init__(self, modelPath, model, X, y, vocsize, ix_to_char, chars, batch_size, epochs):
		self.modelPath = modelPath
		self.model = model
		self.epochs = epochs
		self.X = X
		self.y = y
		self.vocsize = vocsize
		self.ix_to_char = ix_to_char
		self.chars = chars
		self.batch_size = batch_size

	def retrain(self):
		try:
			os.remove(modelPath)
		except Exception:
			pass
		train()
	
	def train(self):
		self.model.fit(self.X, self.y, batch_size=self.batch_size, verbose=1, nb_epoch=self.epochs)
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
