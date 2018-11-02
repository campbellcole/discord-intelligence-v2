import ModelHandler
import Trainer

class TextRNN:

	def __init__(self, DATA_DIR, ALPHA_DIR, MODEL, BATCH_SIZE, HIDDEN_DIM, SEQ_LENGTH, LAYER_NUM, EPOCHS):
		
		X, y, VOCAB_SIZE, ix_to_char, chars = load_data(DATA_DIR, ALPHA_DIR, SEQ_LENGTH)
		self.X = X
		self.y = y
		self.vocsize = VOCAB_SIZE
		self.ix_to_char = ix_to_char
		self.chars = chars

		self.mdh = ModelHandler(HIDDEN_DIM, VOCAB_SIZE, LAYER_NUM, MODEL)
		self.mdl = self.mdh.load()