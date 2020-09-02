### HEADER ###
# HOTSPOT PREDICTION
# description: improve model interpretability by unraveling feature attribution
# input: model, some features
# output: SHAP values
# author: HR

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import shap

pseudocounts = 1
tokPerWindow = 8
embeddingDim = 128


### INPUT ###
# load model
model = tf.keras.models.load_model('CNN/results/model/model.h5')

# load some of the data
tokensAndCounts = pd.read_csv('CNN/data/windowTokens_OPTtraining.csv')
emb = 'CNN/data/embMatrices_OPTtraining.dat'
acc = 'CNN/data/embMatricesAcc_OPTtraining.dat'

def format_input(tokensAndCounts):
    tokens = np.array(tokensAndCounts.loc[:, ['Accession', 'tokens']], dtype='object')
    counts = np.array(tokensAndCounts['counts'], dtype='float32')

    # log-transform counts (+ pseudocounts)
    counts = np.log2((counts + pseudocounts))

    print('number of features: ', counts.shape[0])
    return tokens, counts

def open_and_format_matrices(tokens, counts, emb_path, acc_path):
    no_elements = int(tokPerWindow * embeddingDim)  # number of matrix elements per sliding window
    # how many bytes are this? (32-bit encoding --> 4 bytes per element)
    chunk_size = int(no_elements * 4)

    embMatrix = [None] * tokens.shape[0]
    accMatrix = [None] * tokens.shape[0]
    chunk_pos = 0

    # open weights and accessions binary file
    with open(emb_path, 'rb') as emin, open(acc_path, 'rb') as ain:
        # loop over files to get elements
        for b in range(tokens.shape[0]):
            emin.seek(chunk_pos, 0)  # set cursor position with respect to beginning of file
            # read current chunk of embeddings and format in matrix shape
            dt = np.fromfile(emin, dtype='float32', count=no_elements)

            # make sure to pass 4D-Tensor to model: (batchSize, depth, height, width)
            dt = dt.reshape((tokPerWindow, embeddingDim))
            embMatrix[b] = np.expand_dims(dt, axis=0)

            # get current accession (index)
            ain.seek(int(b * 4), 0)
            accMatrix[b] = int(np.fromfile(ain, dtype='int32', count=1))

            # increment chunk position
            chunk_pos += chunk_size

        emin.close()
        ain.close()

    # order tokens and count according to order in embedding matrix
    accMatrix = np.array(accMatrix, dtype='int32')
    tokens = tokens[accMatrix, :]
    counts = counts[accMatrix]

    embMatrix = np.array(embMatrix, dtype='float32')

    # output: reformatted tokens and counts, embedding matrix
    return tokens, counts, embMatrix


tokens, counts = format_input(tokensAndCounts)
tokens, counts, emb = open_and_format_matrices(tokens, counts, emb, acc)


### MAIN PART ###
background = emb[np.random.choice(emb.shape[0], 100, replace=False), :, :, :]
e = shap.DeepExplainer(model, background)