### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  convert tokens to numerical vectors using a skip-gram neural network
# input:        tokens generated by BPE algorithm in generate_tokens
# output:       word pairs (target and context word)
# author:       HR

print("### PROTEOME EMBEDDING USING SKIP-GRAM NEURAL NETWORK - 1 ###")


import os
import sys
import re
import gc
import numpy as np
import pandas as pd

import threading
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool

import keras
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import text
from keras.preprocessing.text import *

from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Reshape
from keras.layers.embeddings import Embedding
from keras.layers import Dot, concatenate, merge, dot
from keras.layers import *
from keras.engine import input_layer
import tensorflow as tf
from tensorflow.keras import layers
from keras import backend as K

from sklearn.model_selection import train_test_split

# import helper scripts
import skip_gram_NN_helper

gc.enable()

# =============================================================================
# # INPUT
# =============================================================================
words = pd.read_csv(snakemake.input['words'], header = 0)
params = pd.read_csv(snakemake.input['params'], header = 0)


# =============================================================================
# # HYPERPARAMETERS
# =============================================================================

workers = int(params[params['parameter'] == 'threads']['value'])
keep = float(params[params['parameter'] == 'keep']['value'])
negSkipgrams = float(params[params['parameter'] == 'negSkipgrams']['value'])
windowSize = int(params[params['parameter'] == 'windowSize']['value'])

# =============================================================================
# # TEXT PREPROCESSING
# =============================================================================
print("TEXT PREPROCESSING")
# extract tokens
tokens = list(words['tokens'])
tokens = ' '.join(tokens)
tokens = tokens.split(' ')

# functions needed to generate tokens
tokenizer = text.Tokenizer(num_words = len(tokens))
tokenizer.fit_on_texts(tokens)
word2id = tokenizer.word_index
id2word = {v:k for k, v in word2id.items()}


# word-IDs have to be generated for whole sequence data set to prevent duplications
print('CONVERT WORDS TO IDs')
wids = [[word2id[w] for w in text.text_to_word_sequence(token)] for token in words['tokens']]
no_of_sequences = len(wids)

print("number of sequences / batches: {}".format(no_of_sequences))
vocab_size = len(word2id.keys())

print("vocabulary size: {}".format(vocab_size))



print('GENERATE SKIP-GRAMS')
n_batches = no_of_sequences
print("skip-grams are calculated in batches (equivalent to sequences)")


# parallel processing
# generate skip-grams for every sequence independently
pool = multiprocessing.Pool(workers)

if __name__ == "__main__":
    pool.starmap( skip_gram_NN_helper.parallel_processing,
                ([[n, wids[n], windowSize, vocab_size, n_batches, snakemake.output['skip_grams'], keep, negSkipgrams] for n in range(n_batches)]) )
print('done with generating skip-grams')

pool.close()

print("SAVE WORD-IDS")

# save corresponding IDs
ids = pd.DataFrame(word2id.items())
pd.DataFrame.to_csv(ids, snakemake.output['ids'], header=False, index = False)
