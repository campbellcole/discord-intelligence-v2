import numpy as np
from keras.models import Sequential
from Utils import debug

class Trainer():

	def __init__(self, MODEL, model, X, y, VOCAB_SIZE, ix_to_char, chars, BATCH_SIZE):
		self.MODEL = MODEL
		self.model = model
		self.BATCH_SIZE = BATCH_SIZE
		self.X = X
		self.y = y
		self.VOCAB_SIZE = VOCAB_SIZE
		self.ix_to_char = ix_to_char
		self.chars = chars

	def retrain(self, epochs):
		debug('[Trainer]: Removing model...')
		try:
			os.remove(modelPath)
		except Exception:
			pass
		train(epochs)
	
	def train(self, epochs):
		debug('[Trainer]: Training model {} times...'.format(epochs))
		self.model.fit(self.X, self.y, batch_size=self.BATCH_SIZE, verbose=0, epochs=epochs)
		debug('[Trainer]: Saving model...')
		self.model.save(self.MODEL, overwrite=True)

	def generate(self, length, initx):
		debug('[Trainer]: Generating {} characters...'.format(length))
		if initx == None:
			ix = [np.random.randint(self.VOCAB_SIZE)]
		else:
			ix = [initx]
		y_char = [self.ix_to_char[ix[-1]]]
		X = np.zeros((1, length, self.VOCAB_SIZE))
		for i in range(length):
			X[0, i, :][ix[-1]] = 1
			ix = np.argmax(self.model.predict(X[:, :i+1, :])[0], 1)
			y_char.append(self.ix_to_char[ix[-1]])
		return ('').join(y_char)