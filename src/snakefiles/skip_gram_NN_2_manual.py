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
import gc
import numpy as np
import pandas as pd

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

# GPU settings - https://michaelblogscode.wordpress.com/2017/10/10/reducing-and-profiling-gpu-memory-usage-in-keras-with-tensorflow-backend/
# tensorflow wizardy
config = tf.ConfigProto()
config.gpu_options.allow_growth = True # do not pre-allocate memory
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # only allow half of the memory to be allocated
K.tensorflow_backend.set_session(tf.Session(config=config)) # create session

# =============================================================================
# # HYPERPARAMETERS
# =============================================================================
workers = 12

# window of a word: [i - window_size, i + window_size+1]
embeddingDim = 112
epochs = 200

batchSize = 32
valSplit = 0.20

learning_rate = 0.01
adam_decay = 0.005200110247661778

# =============================================================================
# # INPUT
# =============================================================================

# =============================================================================
# # tmp!!
os.chdir('/home/hroetsc/Documents/ProtTransEmbedding/Snakemake')
target_word = np.array(pd.read_csv('results/embedded_proteome/opt_target_10000.txt', delimiter = '\t', names = ['target_word']), dtype='int32')
context_word = np.array(pd.read_csv('results/embedded_proteome/opt_context_10000.txt', delimiter = '\t', names = ['context_word']), dtype='int32')
Y = np.array(pd.read_csv('results/embedded_proteome/opt_label_10000.txt', delimiter = '\t', names = ['label']), dtype='int32')
ids = pd.read_csv('results/embedded_proteome/opt_seq2vec_ids_10000.csv', header = 0)
#
# =============================================================================
print("LOAD DATA")

print('target word vector')
#target_word = np.array(pd.read_csv(snakemake.input['target'], delimiter = '\t', names = ['target_word']), dtype='int32')
target_word = target_word.reshape(target_word.shape[0])
print(target_word)

print('context word vector')
#context_word = np.array(pd.read_csv(snakemake.input['context'], delimiter = '\t', names = ['context_word']), dtype='int32')
context_word = context_word.reshape(context_word.shape[0])
print(context_word)

print('label vector')
#Y = np.array(pd.read_csv(snakemake.input['label'], delimiter = '\t', names = ['label']), dtype='int32')
Y = Y.reshape(Y.shape[0])
print(Y)

#ids = pd.read_csv(snakemake.input['ids'], header = 0)
vocab_size = len(ids.index)+2
print("vocabulary size (number of target word IDs +2): {}".format(vocab_size))

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
output = Dense(1, activation='softmax', kernel_initializer = 'glorot_uniform', name='1st_softmax')(dot_product)
output = Dense(1, activation='softmax', kernel_initializer = 'glorot_uniform', name='2nd_softmax')(output)

# create the primary training model
model = Model(inputs=[input_target, input_context], outputs=output)

adam = Adam(lr=learning_rate, decay=adam_decay)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy']) # binary for binary decisions, categorical for classifications

# view model summary
print(model.summary())

# =============================================================================
# # TRAINING
# =============================================================================
print("MODEL TRAINING")
print('metrics: {}'.format(model.metrics_names))

# USE FIT_GENERATOR
# train on batch - make batch generator threadsafe (with small number of steps and multiprocessing otherwise duplicated batches occur)
# https://stackoverflow.com/questions/56441216/on-fit-generator-and-thread-safety

# train on batch - generate batches
# https://stackoverflow.com/questions/46493419/use-a-generator-for-keras-model-fit-generator
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

# split data into training and validation
print("split word pairs into training and validation data sets")
target_train, target_test, context_train, context_test, Y_train, Y_test = train_test_split(target_word, context_word, Y, test_size=valSplit)

# =============================================================================
# # OLD BATCH GENERATOR APPROACHES
# # iterate systematically
# def batch_generator(target, context, Y, batch_size):
#     n_batches = int(np.ceil(target.shape[0]/int(batch_size))) # divide input length by batch size
#     counter = 0
#     #threading.Lock()
#     while 1:
#         target_batch = target[batch_size*counter:batch_size*(counter+1)]
#         context_batch = context[batch_size*counter:batch_size*(counter+1)]
#         Y_batch = Y[batch_size*counter:batch_size*(counter+1)]
#
#         #print([target_batch, context_batch], Y_batch)
#
#         counter += 1
#
#         yield([target_batch, context_batch], Y_batch)
#
#         if counter >= n_batches: # clean for next epoch
#             counter = 0
#
#         gc.collect()
#
# # use random integers
# # does not work properly at the moment bc random integers are always the same
# def batch_generator2(target, context, Y, batch_size):
#     counter = 0
#     while True:
#         idx = np.array(np.random.randint(0, (target.shape[0])-1, size = batch_size), dtype = 'int32')
#
#         target_batch = target[idx]
#         context_batch = context[idx]
#         Y_batch = Y[idx]
#
#         counter += 1
#
#         #print([target_batch, context_batch], Y_batch)
#         yield ([target_batch, context_batch], Y_batch)
#
#         gc.collect()
#
# =============================================================================

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
         batch_target = self.target[idx*self.batch_size : (idx + 1)*self.batch_size]
         batch_context = self.context[idx*self.batch_size : (idx + 1)*self.batch_size]
         batch_Y = self.Y[idx*self.batch_size : (idx + 1)*self.batch_size]

         return [np.array(batch_target, dtype = 'int32'), np.array(batch_context, dtype = 'int32')], np.array(batch_Y, dtype = 'int32')

    #def on_epoch_end(self):
        #pass


# apply batch generator
print("generating batches for model training")
train_generator = BatchGenerator(target_train, context_train, Y_train, batchSize)
test_generator = BatchGenerator(target_test, context_test, Y_test, batchSize)


# fit model
print("fit the model")

# can be ignored in case batch generator uses keras.utils.Sequence() class
steps = np.ceil((target_train.shape[0]/batchSize)*0.1)
val_steps = np.ceil((target_test.shape[0]/batchSize)*0.1)

fit = model.fit_generator(generator=train_generator,
                    validation_data=test_generator,
                    steps_per_epoch = steps,
                    validation_steps = val_steps,
                    epochs = epochs,
                    initial_epoch=0,
                    verbose=2,
                    workers=workers,
                    use_multiprocessing=True,
                    shuffle=True)


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
model.save('results/embedded_proteome/opt_model_10000.csv')

# save accuracy and loss
m = open('results/embedded_proteome/opt_model_metrics_1000.txt', 'w')
m.write("accuracy \t {} \n val_accuracy \t {} \n loss \t {} \n val_loss \t {}".format(fit.history['acc'], fit.history['val_acc'], fit.history['loss'], fit.history['val_loss']))
m.close()

K.clear_session()
tf.reset_default_graph()
