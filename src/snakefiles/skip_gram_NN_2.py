### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  convert tokens to numerical vectors using a skip-gram neural network
# input:        word pairs (target and context word) generated in skip_gram_NN_1
# output:       embedded tokens (weights and their IDs)
# author:       HR

print("### PROTEOME EMBEDDING USING SKIP-GRAM NEURAL NETWORK - 2 ###")
import os
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

gc.enable()

# GPU settings - https://michaelblogscode.wordpress.com/2017/10/10/reducing-and-profiling-gpu-memory-usage-in-keras-with-tensorflow-backend/
# tensorflow wizardy
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True # do not pre-allocate memory
#config.gpu_options.per_process_gpu_memory_fraction = 0.5 # only allow half of the memory to be allocated
#K.tensorflow_backend.set_session(tf.Session(config=config)) # create session

# HYPERPARAMETERS
workers = 12

# window of a word: [i - window_size, i + window_size+1]
embeddingDim = 100
epochs = 200

batchSize = 32
valSplit = 0.20

# INPUT
print("LOAD DATA")

print("target word vector")
target_word = np.array(pd.read_csv(snakemake.input['target'], delimiter = '\t', names = ['target_word']), dtype='int32')
target_word = target_word.reshape(target_word.shape[0],1)
print(target_word, target_word.shape)

print("context word vector")
context_word = np.array(pd.read_csv(snakemake.input['context'], delimiter = '\t', names = ['context_word']), dtype='int32')
context_word = context_word.reshape(context_word.shape[0],1)
print(context_word, context_word.shape)

print("label vector")
Y = np.array(pd.read_csv(snakemake.input['label'], delimiter = '\t', names = ['label']), dtype='int32')
Y = Y.reshape(Y.shape[0],1)
print(Y, Y.shape)

print("word ID table")
ids = pd.read_csv(snakemake.input['ids'], header = 0)

# vocab size = number of unique tokens
vocab_size = len(ids.index)+1
print("vocabulary size (number of target word IDs +1): {}".format(vocab_size))


# MODEL CREATION
print("MODEL GENERATION")

# model - https://adventuresinmachinelearning.com/word2vec-keras-tutorial/
# https://adventuresinmachinelearning.com/word2vec-tutorial-tensorflow/
input_target = Input(((1,)), name='target_word')
input_context = Input(((1,)), name='context_word')

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

# create the primary training model
model = Model(inputs=[input_target, input_context], outputs=output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # binary for binary decisions, categorical for classifications

# view model summary
print(model.summary())

# TRAINING
print("MODEL TRAINING")
print('metrics: {}'.format(model.metrics_names))

# USE TRAIN_ON_BATCH
#target = np.zeros((1,))
#context = np.zeros((1,))
#Y_label = np.zeros((1,))

#for r in range(epochs):
#    idx = np.random.randint(0, (Y.shape[0])-1, size = batchSize)
#    target = target_word[idx]
#    context = context_word[idx]
#    Y_label = Y[idx]
#
#    fit = model.train_on_batch([target, context], Y_label,
#                                reset_metrics = False)

#    print('epoch: {} - loss: {} - accuracy: {}'.format(r, fit[0], fit[1]))
#
#    target = np.zeros((1,))
#    context = np.zeros((1,))
#    Y_label = np.zeros((1,))
#
#    gc.collect()

# USE FIT_GENERATOR
# train on batch - make batch generator threadsafe (with small number of steps and multiprocessing otherwise duplicated batches occur)
# https://stackoverflow.com/questions/56441216/on-fit-generator-and-thread-safety

# train on batch - generate batches
# https://stackoverflow.com/questions/46493419/use-a-generator-for-keras-model-fit-generator
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly


def batch_generator(target, context, Y, batch_size):
    n_batches = int(np.ceil(target.shape[0]/int(batch_size))) # divide input length by batch size
    counter = 0
    #threading.Lock()
    while 1:
        target_batch = np.array(target[batch_size*counter:batch_size*(counter+1)], dtype='int32').reshape(batch_size,1)
        context_batch = np.array(context[batch_size*counter:batch_size*(counter+1)], dtype='int32').reshape(batch_size,1)
        Y_batch = np.array(Y[batch_size*counter:batch_size*(counter+1)], dtype='int32').reshape(batch_size,1)

        counter += 1
        return([target_batch, context_batch], Y_batch)

        if counter >= n_batches: # clean for next epoch
            counter = 0
    gc.collect()

# split data into training and validation

print("split word pairs into training and validation data sets")
target_train, target_test, context_train, context_test, Y_train, Y_test = train_test_split(target_word, context_word, Y, test_size=valSplit)

# apply batch generator
print("generating batches for model training")
train_generator = batch_generator(target_train, context_train, Y_train, batchSize)
test_generator = batch_generator(target_test, context_test, Y_test, batchSize)

# fit model
print("fit the model")

steps = np.ceil(target_train.shape[0]/batchSize)
val_steps = np.ceil(target_test.shape[0]/batchSize)

fit = model.fit_generator(generator=train_generator,
                    validation_data=test_generator,
                    steps_per_epoch = steps,
                    validation_steps = val_steps,
                    epochs=epochs,
                    initial_epoch=0,
                    verbose=2,
                    workers=workers,
                    use_multiprocessing=True,
                    shuffle=True)


# USE FIT FUNCTION
#fit = model.fit(x = [target_train, context_train],
#                y = Y_train,
#                batch_size = batchSize,
#                epochs = epochs,
#                verbose = 2,
#                validation_split = valSplit,
#                validation_data = [[target_test, context_test], Y_test],
#                steps_per_epoch = steps,
#                validation_steps = val_steps,
#                initial_epoch=0,
#                workers=workers,
#                use_multiprocessing=True,
#                shuffle=True)


print("SAVE WEIGHTS")
### OUTPUT ###
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
m.write("accuracy \t {} \n val_accuracy \t {} \n loss \t {} \n val_loss \t {}".format(fit.history['acc'], fit.history['val_acc'], fit.history['loss'], fit.history['val_loss']))
m.close()

#m = open(snakemake.output['metrics'], 'w')
#m.write("accuracy \t {} \n loss \t {}".format(fit[1], fit[0]))
#m.close()

K.clear_session()
tf.reset_default_graph()
