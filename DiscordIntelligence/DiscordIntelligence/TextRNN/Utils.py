from __future__ import print_function
import numpy as np
import re

def load_data(data_dir, alphabet_dir, seq_length):
	data = open(data_dir, 'r', encoding="utf8").read()
	chars = open(alphabet_dir, 'r', encoding="utf8").read()
	chars = list(set(chars))
	chars.sort()
	VOCAB_SIZE = len(chars)

	data = ''.join([ch for ch in data if ch in chars])

	#print('Data length: {} characters'.format(len(data)))
	#print('Vocabulary size: {} characters'.format(VOCAB_SIZE))

	ix_to_char = {ix:char for ix, char in enumerate(chars)}
	char_to_ix = {char:ix for ix, char in enumerate(chars)}

	seg = int(len(data)/seq_length)

	X = np.zeros((seg, seq_length, VOCAB_SIZE))
	y = np.zeros((seg, seq_length, VOCAB_SIZE))

	for i in range(0, seg):
		X_sequence = data[i*seq_length:(i+1)*seq_length]
		X_sequence_ix = [char_to_ix[value] for value in X_sequence]
		input_sequence = np.zeros((seq_length, VOCAB_SIZE))
		for j in range(seq_length):
			input_sequence[j][X_sequence_ix[j]] = 1.
			X[i] = input_sequence

		y_sequence = data[i*seq_length+1:(i+1)*seq_length+1]
		y_sequence_ix = [char_to_ix[value] for value in y_sequence]
		target_sequence = np.zeros((seq_length, VOCAB_SIZE))
		for j in range(seq_length):
			target_sequence[j][y_sequence_ix[j]] = 1.
			y[i] = target_sequence
	return X, y, VOCAB_SIZE, ix_to_char, chars