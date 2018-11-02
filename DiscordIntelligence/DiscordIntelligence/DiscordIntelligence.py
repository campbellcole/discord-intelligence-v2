import argparse
import os
import discord
import asyncio

import sys
sys.path.append('./TextRNN')

from Trainer import Trainer
from ModelHandler import ModelHandler
from Utils import *

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
debug('Locating model...')
mdh = ModelHandler(MODEL)
mdl = None
if os.path.isfile(MODEL):
	debug('Found model. Loading...')
	mdl = mdh.load()
else:
	debug("Couldn't find model. Creating...")
	mdl = mdh.create(HIDDEN_DIM, VOCAB_SIZE, LAYER_NUM)

debug('Creating trainer...')
trainer = Trainer(MODEL, mdl, DATA_DIR, ALPHA_DIR)

# discord
client = discord.Client()

@client.event
async def on_ready():
	debug("Connected to Discord.")

@client.event
async def on_message(message):
	if message.content.startswith(PREFIX) and not message.author == client.user:
		debug('Got message: "{}"'.format(message.content)) # remove this later it will cause incredible spam
		msg = await client.send_message(message.channel, 'Processing request...')
		cmd = message.content[len(PREFIX):].strip()
		spl = cmd.split()
		if spl[0] == 'generate':
			len = 10
			initx = None
			try:
				len = int(spl[1])
				initx = int(spl[2])
			except Exception:
				pass
			await client.edit_message(msg, 'Generating...')
			gen = trainer.generate(None, 10)
			await client.edit_message(msg, gen)

debug('Attempting login with token {}'.format(TOKEN))
client.run(TOKEN)

debug('Exiting...')