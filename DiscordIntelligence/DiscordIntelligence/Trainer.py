import threading
import os
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
import ModelHandler

class Trainer(threading.Thread):

	def __init__(self, modelPath, model, X, y, vocsize, ix_to_char, chars, batch_size = 50, epochs = 50):
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
		ep = 0
		while True:
			print('\n\nEpochs: {}\n'.format(ep))
			model.fit(self.X, self.y, batch_size=self.batch_size, verbose=1, nb_epoch=1)
			model.save(modelPath, overwrite=True)
			ep+=1
			if ep == epochs:
				exit()