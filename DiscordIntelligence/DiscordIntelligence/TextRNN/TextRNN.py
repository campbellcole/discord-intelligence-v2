from ModelHandler import ModelHandler
from Trainer import Trainer
from Utils import *

class TextRNN:

	def __init__(self, DATA_DIR, ALPHA_DIR, MODEL, BATCH_SIZE, HIDDEN_DIM, SEQ_LENGTH, LAYER_NUM, DBG=False):
		
		set_debug(DBG)

		debug('[TextRNN]: Loading data...')
		X, y, VOCAB_SIZE, ix_to_char, chars = load_data(DATA_DIR, ALPHA_DIR, SEQ_LENGTH)

		debug('[TextRNN]: Creating ModelHandler...')
		self.modelhandler = ModelHandler(HIDDEN_DIM, VOCAB_SIZE, LAYER_NUM, MODEL)
		debug('[TextRNN]: Loading model...')
		self.model = self.modelhandler.load()

		debug('[TextRNN]: Creating Trainer...')
		self.trainer = Trainer(MODEL, self.model, X, y, VOCAB_SIZE, ix_to_char, chars, BATCH_SIZE)

	def train(self, epochs = 50):
		debug('[TextRNN]: Training {} times...'.format(epochs))
		self.trainer.train(epochs)

	def generate(self, length, initx):
		debug('[TextRNN]: Generating {} characters...'.format(length))
		return self.trainer.generate(length, initx)