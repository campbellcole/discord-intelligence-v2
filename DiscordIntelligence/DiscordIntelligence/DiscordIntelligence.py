import argparse
import os
import discord
import asyncio

import sys
sys.path.append('./TextRNN')

from TextRNN import TextRNN

ap = argparse.ArgumentParser();
ap.add_argument('--data-dir', default='res\\study-data.txt', help='file to read study data from')
ap.add_argument('--alpha-dir', default='res\\alphabet.txt', help='file containing alphabet to use')
ap.add_argument('--model', default='res\\network.hdf5', help='path to model to load')
ap.add_argument('--token', help='discord client secret key')
ap.add_argument('--prefix', default='net: ', help='prefix for discord commands')
ap.add_argument('--debug', action='store_true', help='debug/verbose mode')
args = vars(ap.parse_args())

DATA_DIR = args['data_dir']
ALPHA_DIR = args['alpha_dir']
MODEL = args['model']
TOKEN = args['token']
PREFIX = args['prefix']
DEBUG = args['debug']

BATCH_SIZE = 50
HIDDEN_DIM = 500
SEQ_LENGTH = 50
LAYER_NUM = 2
EPOCHS = 50

def debug(out):
	if DEBUG:
		print(out)

# initialize
debug("Creating TextRNN...")
net = TextRNN(DATA_DIR, ALPHA_DIR, MODEL, BATCH_SIZE, HIDDEN_DIM, SEQ_LENGTH, LAYER_NUM, DEBUG)

# discord
client = discord.Client()

@client.event
async def on_ready():
	print("Ready.")

@client.event
async def on_message(message):
	if message.author == client.user:
		return
	open(DATA_DIR, 'a', encoding='utf8').write(message.content+'\n')
	debug('Got message: "{}"'.format(message.content))
	if message.content.startswith(PREFIX):
		msg = await client.send_message(message.channel, 'Processing request...')
		cmd = message.content[len(PREFIX):].strip()
		spl = cmd.split()
		if spl[0] == 'generate':
			length = 10
			initx = None
			try:
				length = int(spl[1])
				initx = int(spl[2])
			except Exception:
				pass
			await client.edit_message(msg, 'Generating {} characters...'.format(length))
			gen = net.generate(length, None)
			await client.edit_message(msg, gen)
		if spl[0] == 'train':
			epochs = 10
			try:
				epochs = int(spl[1])
			except Exception:
				pass
			await client.edit_message(msg, 'Training {} times...'.format(epochs))
			net.train(epochs)
			await client.edit_message(msg, 'Done training...')
		if spl[0] == 'retrain':
			epochs = 10
			try:
				epochs = int(spl[1])
			except Exception:
				pass
			await client.edit_message(msg, 'Deleting old model and training {} times...'.format(epochs))
			net.retrain(epochs)

debug('Attempting login with token {}'.format(TOKEN))
client.run(TOKEN)

debug('Exiting...')