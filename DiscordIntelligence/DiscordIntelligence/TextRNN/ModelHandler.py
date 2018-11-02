import os
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed

class ModelHandler:

	def __init__(self, MODEL = None):
		self.mdl = MODEL

	def load(self):
		if self.mdl == None:
			raise ValueError('No model given for loading')
		m = load_model(self.mdl)
		if m == None:
			raise WindowsError('Unable to load model')
		return m

	def create(self, HIDDEN_DIM, VOCAB_SIZE, LAYER_NUM):
		t = Sequential()
		t.add(LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequences=True))
		for i in range(LAYER_NUM - 1):
			t.add(LSTM(HIDDEN_DIM, return_sequences=True))
		t.add(TimeDistributed(Dense(VOCAB_SIZE)))
		t.add(Activation('softmax'))
		t.compile(loss="categorical_crossentropy", optimizer="rmsprop")
		return t