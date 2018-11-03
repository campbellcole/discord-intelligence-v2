from __future__ import print_function
import numpy as np
import re

def set_debug(dbg):
	global DEBUG
	DEBUG = dbg

def debug(out):
	if DEBUG:
		print(out)

def load_data(data_dir, alphabet_dir, seq_length):
	debug('[Utils]: Reading study data...')
	data = open(data_dir, 'r', encoding="utf8").read()
	debug('[Utils]: {} characters.'.format(len(data)))
	debug('[Utils]: Reading alphabet...')
	chars = open(alphabet_dir, 'r', encoding="utf8").read()
	debug('[Utils]: {} characters.'.format(len(chars)))
	debug('[Utils]: Converting alphabet to list...')
	chars = list(set(chars))
	debug('[Utils]: Sorting alphabet...')
	chars.sort()
	VOCAB_SIZE = len(chars)

	debug('[Utils]: Removing duplicate characters from alphabet...')
	data = ''.join([ch for ch in data if ch in chars])

	debug('[Utils]: Enumerating Index->Character')
	ix_to_char = {ix:char for ix, char in enumerate(chars)}
	debug('[Utils]: Enumerating Character->Index')
	char_to_ix = {char:ix for ix, char in enumerate(chars)}

	seg = int(len(data)/seq_length)

	debug('[Utils]: Creating empty arrays...')
	X = np.zeros((seg, seq_length, VOCAB_SIZE))
	y = np.zeros((seg, seq_length, VOCAB_SIZE))

	debug('[Utils]: Populating arrays with study data...')
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