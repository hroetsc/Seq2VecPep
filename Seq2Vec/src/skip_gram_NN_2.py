### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  convert tokens to numerical vectors using a skip-gram neural network
# input:        word pairs (target and context word) generated in skip_gram_NN_1
# output:       embedded tokens (weights and their IDs)
# author:       HR

print("### PROTEOME EMBEDDING USING SKIP-GRAM NEURAL NETWORK - 2 ###")

import os
import sys
import gc
import threading
import numpy as np
import pandas as pd

import argparse

import keras

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

from keras.utils import Sequence
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

gc.enable()


# =============================================================================
# # INPUT
# =============================================================================
print("LOAD DATA")
params = pd.read_csv(snakemake.input['params'], header = 0)
skip_grams = pd.read_csv(snakemake.input['skip_grams'], sep = " ", header = None)
ids = pd.read_csv(snakemake.input['ids'], header = None)

# =============================================================================
# # HYPERPARAMETERS
# =============================================================================

workers = int(params[params['parameter'] == 'threads']['value'])

embeddingDim = int(params[params['parameter'] == 'embedding']['value'])
epochs = int(params[params['parameter'] == 'epochs']['value'])

valSplit = float(params[params['parameter'] == 'valSplit']['value'])
batchSize = int(params[params['parameter'] == 'batchSize']['value'])

seqtype = params[params['parameter'] == 'Seqtype']['value']

if seqtype == 'AA':
    learning_rate = 0.004
    adam_decay = 1e-06
elif seqtype == 'RNA':
    learning_rate = 0.004
    adam_decay = 1e-06


# =============================================================================
# split skip-grams into target, context and label np.array()
target_word = np.array(skip_grams.iloc[:,0], dtype = 'int32')
context_word = np.array(skip_grams.iloc[:,1], dtype = 'int32')
Y = np.array(skip_grams.iloc[:,2], dtype = 'int32')

print('target word vector')
target_word = target_word.reshape(target_word.shape[0],)
print(target_word)

print('context word vector')
context_word = context_word.reshape(context_word.shape[0],)
print(context_word)

print('label vector (converted 0 to -1)')
Y = Y.reshape(Y.shape[0],)
# replace 0 by -1
Y = np.where(Y == 0, -1, Y)
print(Y)

vocab_size = len(ids.index) + 1
print("vocabulary size (number of target word IDs + 1): {}".format(vocab_size))


# =============================================================================
# # MODEL CREATION
# =============================================================================
print("MODEL GENERATION")

input_target = keras.layers.Input(((1,)), name='target_word')
input_context = keras.layers.Input(((1,)), name='context_word')

# embed input layers
embedding = Embedding(input_dim = vocab_size,
                        output_dim = embeddingDim,
                        input_length = 1,
                        embeddings_initializer = 'he_uniform',
                        name = 'embedding')

# apply embedding
target = embedding(input_target)
target = Reshape((embeddingDim,1), name='target_embedding')(target) # every individual skip-gram has dimension embedding x 1
context = embedding(input_context)
context = Reshape((embeddingDim,1), name='context_embedding')(context)

# dot product similarity - normalize to get value between 0 and 1!
dot_product = dot([target, context], axes = 1, normalize = True, name = 'dot_product')
dot_product = Reshape((1,))(dot_product)

# add dense layers
output = Dense(64, activation = 'tanh', kernel_initializer = 'he_uniform', name='1st_dense')(dot_product)
output = Dropout(0.5)(output)
output = Dense(1, activation = 'tanh', kernel_initializer = 'he_uniform', name='2nd_dense')(output)


# create the primary training model
model = Model(inputs=[input_target, input_context], outputs=output)

adam = Adam(lr=learning_rate, decay=adam_decay)
model.compile(loss='squared_hinge', optimizer=adam, metrics=['accuracy'])

# view model summary
print(model.summary())

# =============================================================================
# # TRAINING
# =============================================================================
print("MODEL TRAINING")
# split data into training and validation
print("split word pairs into training and validation data sets")
target_train, target_test, context_train, context_test, Y_train, Y_test = train_test_split(target_word, context_word, Y, test_size=valSplit)

print('model metrics: {}'.format(model.metrics_names))


class BatchGenerator(keras.utils.Sequence):

     def __init__(self, target, context, Y, batch_size):
         self.target, self.context, self.Y = target, context, Y
         self.batch_size = batch_size

     def __len__(self):
         return int(np.ceil(len(self.target) / float(self.batch_size)))

     def __getitem__(self, idx):

         batch_target = self.target[idx*self.batch_size : (idx + 1)*self.batch_size]
         batch_context = self.context[idx*self.batch_size : (idx + 1)*self.batch_size]
         batch_Y = self.Y[idx*self.batch_size : (idx + 1)*self.batch_size]

         return [batch_target, batch_context], batch_Y

     def on_epoch_end(self):
         pass

# apply batch generator
print("generating batches for model training")
train_generator = BatchGenerator(target_train, context_train, Y_train, batchSize)
test_generator = BatchGenerator(target_test, context_test, Y_test, batchSize)

# fit model
print("fit the model")

fit = model.fit_generator(generator = train_generator,
                    validation_data = test_generator,
                    epochs = epochs,
                    initial_epoch = 0,
                    verbose=2,
                    max_queue_size = 1,
                    workers = workers,
                    use_multiprocessing = False,
                    shuffle = False)

# shuffle has to be false bc BatchBenerator can't cope with shuffled data!

# =============================================================================
# ### OUTPUT ###
# =============================================================================
print("SAVE WEIGHTS")
# get word embedding
print("configuration of embedding layer:")
print(model.layers[2].get_config())
weights = model.layers[2].get_weights()[0] # weights of the embedding layer of target word

# save weights of embedding matrix
df = pd.DataFrame(weights)
pd.DataFrame.to_csv(df, snakemake.output['weights'], header=False)
df.head()

# save model
model.save(snakemake.output['model'])

# save accuracy and loss
m = open(snakemake.output['metrics'], 'w')
m.write("accuracy \t {} \n val_accuracy \t {} \n loss \t {} \n val_loss \t {}".format(fit.history['accuracy'], fit.history['val_accuracy'], fit.history['loss'], fit.history['val_loss']))
m.close()

K.clear_session()
tf.compat.v1.reset_default_graph()
