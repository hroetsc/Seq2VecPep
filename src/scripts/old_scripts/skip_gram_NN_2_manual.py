#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:55:50 2020

@author: hroetsc
"""


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
import numpy as np
import pandas as pd
#import math

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

print("cleaning tensorflow session")
K.clear_session()
tf.reset_default_graph()

# GPU settings - https://michaelblogscode.wordpress.com/2017/10/10/reducing-and-profiling-gpu-memory-usage-in-keras-with-tensorflow-backend/
# tensorflow wizardy
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True # do not pre-allocate memory
#config.gpu_options.per_process_gpu_memory_fraction = 0.5 # only allow half of the memory to be allocated
#K.tensorflow_backend.set_session(tf.Session(config=config)) # create session

# =============================================================================
# # HYPERPARAMETERS
# =============================================================================
workers = 16

# window of a word: [i - window_size, i + window_size+1]
embeddingDim = 100
epochs = 50

batchSize = 32
valSplit = 0.20

learning_rate = 0.01
adam_decay = 0.005200110247661778

# =============================================================================
# # INPUT
# =============================================================================
print("LOAD DATA")

# =============================================================================
# # tmp!!
os.chdir('/home/hanna/Documents/QuantSysBios/ProtTransEmbedding/Snakemake')
skip_grams = pd.DataFrame(pd.read_csv("results/embedded_proteome/opt_skipgrams_reduced_10000.csv", header = 0))
ids = pd.read_csv('results/embedded_proteome/opt_seq2vec_ids_10000.csv', header = 0)
#
# =============================================================================
# split skip-grams into target, context and label np.array()
target_word = np.array(skip_grams.iloc[:,0], dtype = 'int32')
context_word = np.array(skip_grams.iloc[:,1], dtype = 'int32')
Y = np.array(skip_grams.iloc[:,2], dtype = 'int32')

print('target word vector')
target_word = target_word.reshape(target_word.shape[0],1)
print(target_word)

print('context word vector')
context_word = context_word.reshape(context_word.shape[0],1)
print(context_word)

print('label vector')
Y = Y.reshape(Y.shape[0],1)
print(Y)

vocab_size = len(ids.index)+2
print("vocabulary size (number of target word IDs +2): {}".format(vocab_size))

### remove!!! ### just for speed testing
ind = np.array(np.random.randint(0, target_word.shape[0], size = 200000), dtype = 'int32')
target_word = target_word[ind]
context_word = context_word[ind]
Y = Y[ind]

# =============================================================================
# # MODEL CREATION
# =============================================================================
print("MODEL GENERATION")

# model - https://adventuresinmachinelearning.com/word2vec-keras-tutorial/
# https://adventuresinmachinelearning.com/word2vec-tutorial-tensorflow/
input_target = keras.layers.Input(((1,)), name='target_word')
input_context = keras.layers.Input(((1,)), name='context_word')

# embed input layers
# https://github.com/keras-team/keras/issues/3110
embedding = Embedding(input_dim = vocab_size,
                        output_dim = embeddingDim,
                        input_length = 1,
                        embeddings_initializer = 'glorot_uniform',
                        name = 'embedding')
# later create lookup table with weights so that one could initialize the embedding layer with pretrained weights

# apply embedding
target = embedding(input_target)
target = Reshape((embeddingDim,1), name='target_embedding')(target) # every individual skip-gram has dimension embedding x 1
context = embedding(input_context)
context = Reshape((embeddingDim,1), name='context_embedding')(context)

# dot product similarity - normalize to get value between 0 and 1!
dot_product = dot([target, context], axes = 1, normalize = True, name = 'dot_product')
dot_product = Reshape((1,))(dot_product)

# add the sigmoid dense layer
output = Dense(1, activation='sigmoid', kernel_initializer = 'glorot_uniform', name='1st_sigmoid')(dot_product)
output = Dense(1, activation='sigmoid', kernel_initializer = 'glorot_uniform', name='2nd_sigmoid')(output)
output = Dense(1, activation='sigmoid', kernel_initializer = 'glorot_uniform', name='3rd_sigmoid')(output)
output = Dense(1, activation='sigmoid', kernel_initializer = 'glorot_uniform', name='4th_sigmoid')(output)
output = Dense(1, activation='sigmoid', kernel_initializer = 'glorot_uniform', name='5th_sigmoid')(output)
output = Dense(1, activation='sigmoid', kernel_initializer = 'glorot_uniform', name='6th_sigmoid')(output)
output = Dense(1, activation='sigmoid', kernel_initializer = 'glorot_uniform', name='7th_sigmoid')(output)
output = Dense(1, activation='sigmoid', kernel_initializer = 'glorot_uniform', name='8th_sigmoid')(output)
output = Dense(1, activation='sigmoid', kernel_initializer = 'glorot_uniform', name='9th_sigmoid')(output)
output = Dense(1, activation='sigmoid', kernel_initializer = 'glorot_uniform', name='10th_sigmoid')(output)

# create the primary training model
model = Model(inputs=[input_target, input_context], outputs=output)

#adam = Adam(lr=learning_rate, decay=adam_decay)
#model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy']) # binary for binary decisions, categorical for classifications

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy']) # binary for binary decisions, categorical for classifications
# view model summary
print(model.summary())

# =============================================================================
# # TRAINING
# =============================================================================
print("MODEL TRAINING")
# split data into training and validation
print("split word pairs into training and validation data sets")
target_train, target_test, context_train, context_test, Y_train, Y_test = train_test_split(target_word, context_word, Y, test_size=valSplit)

print('metrics: {}'.format(model.metrics_names))

# USE FIT_GENERATOR
# train on batch - make batch generator threadsafe (with small number of steps and multiprocessing otherwise duplicated batches occur)
# https://stackoverflow.com/questions/56441216/on-fit-generator-and-thread-safety

# train on batch - generate batches
# https://stackoverflow.com/questions/46493419/use-a-generator-for-keras-model-fit-generator
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

# # OLD BATCH GENERATOR APPROACHES
# # iterate systematically
def batch_generator(target, context, Y, batch_size):
    n_batches = int(np.ceil(target.shape[0]/int(batch_size))) # divide input length by batch size
    counter = 0
    #threading.Lock()
    while 1:
        target_batch = target[batch_size*counter:batch_size*(counter+1)]
        context_batch = context[batch_size*counter:batch_size*(counter+1)]
        Y_batch = Y[batch_size*counter:batch_size*(counter+1)]

        #print([target_batch, context_batch], Y_batch)

        counter += 1
        yield([target_batch, context_batch], Y_batch)

        if counter >= n_batches: # clean for next epoch
            counter = 0

        gc.collect()

# # use random integers
def batch_generator2(target, context, Y, batch_size):
    counter = 0
    while True:
        idx = np.array(np.random.randint(0, (target.shape[0])-1, size = batch_size), dtype = 'int32')

        target_batch = target[idx]
        context_batch = context[idx]
        Y_batch = Y[idx]

        counter += 1

        #print([target_batch, context_batch], Y_batch)
        yield ([target_batch, context_batch], Y_batch)

        gc.collect()

# make batch generator suitable for multiprocessing - use keras.utils.Sequence class
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence

class BatchGenerator(keras.utils.Sequence):

     def __init__(self, target, context, Y, batch_size):
         self.target, self.context, self.Y = target, context, Y
         self.batch_size = batch_size

     def __len__(self):
         return int(np.ceil(len(self.target) / float(self.batch_size)))

     def __getitem__(self, idx):
         batch_target = np.array(self.target[idx*self.batch_size : (idx + 1)*self.batch_size], dtype = 'int32').reshape(self.batch_size,1)
         batch_context = np.array(self.context[idx*self.batch_size : (idx + 1)*self.batch_size], dtype = 'int32').reshape(self.batch_size,1)
         batch_Y = np.array(self.Y[idx*self.batch_size : (idx + 1)*self.batch_size], dtype = 'int32').reshape(self.batch_size,1)

         return [batch_target, batch_context], batch_Y

     #def on_epoch_end(self):
     #    pass


# apply batch generator
print("generating batches for model training")
train_generator = BatchGenerator(target_train, context_train, Y_train, batchSize)
test_generator = BatchGenerator(target_test, context_test, Y_test, batchSize)

# fit model
print("fit the model")

# can be ignored in case batch generator uses keras.utils.Sequence() class (?)
steps = np.ceil(target_train.shape[0]/batchSize)
val_steps = np.ceil(target_test.shape[0]/batchSize)

fit = model.fit_generator(generator=train_generator,
                    validation_data=test_generator,
                    steps_per_epoch = steps,
                    validation_steps = val_steps,
                    epochs = epochs,
                    initial_epoch = 1,
                    verbose=2,
                    workers=workers,
                    use_multiprocessing=True,
                    shuffle=False)
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
pd.DataFrame.to_csv(df, 'results/embedded_proteome/opt_seq2vec_weights_10000.csv', header=False)
df.head()

# save model
model.save('results/embedded_proteome/opt_model_10000.h5')

# save accuracy and loss
m = open('results/embedded_proteome/opt_model_metrics_1000.txt', 'w')
m.write("accuracy \t {} \n val_accuracy \t {} \n loss \t {} \n val_loss \t {}".format(fit.history['accuracy'], fit.history['val_accuracy'], fit.history['loss'], fit.history['val_loss']))
m.close()

K.clear_session()
tf.reset_default_graph()
