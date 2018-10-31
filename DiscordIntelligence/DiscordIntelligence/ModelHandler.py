import os
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed

class ModelHandler(object):

	def __init__(self, HIDDEN_DIM, VOCAB_SIZE, LAYER_NUM, MODEL = None):
		self.HIDDEN_DIM = HIDDEN_DIM
		self.VOCAB_SIZE = VOCAB_SIZE
		self.LAYER_NUM = LAYER_NUM
		self.mdl = MODEL

	def load():
		return load_model(self.mdl)

	def create(self):
		t = Sequential()
		t.add(LSTM(self.HIDDEN_DIM, input_shape=(None, self.VOCAB_SIZE), return_sequences=True))
		for i in range(self.LAYER_NUM - 1):
			t.add(LSTM(self.HIDDEN_DIM, return_sequences=True))
		t.add(TimeDistributed(Dense(VOCAB_SIZE)))
		t.add(Activation('softmax'))
		t.compile(optimizer="rmsprop", loss="categorial_crossentropy")
		return t