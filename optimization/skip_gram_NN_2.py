### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  convert tokens to numerical vectors using a skip-gram neural network
# input:        word pairs (target and context word) generated in skip_gram_NN_1
# output:       embedded tokens (weights and their IDs)
# author:       HR

print("### PROTEOME EMBEDDING USING SKIP-GRAM NEURAL NETWORK - 2 ###")
import os
import sys
import re
import gc
import numpy as np
import pandas as pd

import threading
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

# import scripts for hyperparameter optimization
import skopt
from skopt import gbrt_minimize, gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer
from keras.optimizers import Adam

gc.enable()

# GPU settings - https://michaelblogscode.wordpress.com/2017/10/10/reducing-and-profiling-gpu-memory-usage-in-keras-with-tensorflow-backend/
# tensorflow wizardy
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True # do not pre-allocate memory
#config.gpu_options.per_process_gpu_memory_fraction = 0.5 # only allow half of the memory to be allocated
#K.tensorflow_backend.set_session(tf.Session(config=config)) # create session

os.chdir('/home/hanna/Documents/QuantSysBios/ProtTransEmbedding/Snakemake')

# HYPERPARAMETERS
workers = 16

# window of a word: [i - window_size, i + window_size+1]
embeddingDim = 5 # used as maximum by Mikolov et al. 2013 for NLP word embedding
epochs = 5 #100 #200

batchSize = 64
valSplit = 0.20

# INPUT
size = sys.argv[1]
print("current vocabulary size: {}".format(size))

print("LOAD DATA")

print('target word vector')
target_word = np.array(pd.read_csv('results/embedded_proteome/opt_target_{}.txt'.format(size), delimiter = '\t', names = ['target_word']), dtype='int32')
target_word = target_word.reshape(target_word.shape[0])
print(target_word)

print('context word vector')
context_word = np.array(pd.read_csv('results/embedded_proteome/opt_context_{}.txt'.format(size), delimiter = '\t', names = ['context_word']), dtype='int32')
context_word = context_word.reshape(context_word.shape[0])
print(context_word)

print('label vector')
Y = np.array(pd.read_csv('results/embedded_proteome/opt_label_{}.txt'.format(size), delimiter = '\t', names = ['label']), dtype='int32')
Y = Y.reshape(Y.shape[0])
print(Y)

ids = pd.read_csv('results/embedded_proteome/opt_seq2vec_ids_{}.csv'.format(size), header = 0)
vocab_size = len(ids.index)+2
print("vocabulary size (number of target word IDs +1): {}".format(vocab_size))

#no_skip_grams = int(target_word.shape[0])*0.1
# for the sake of speed, pick random 10 % of original skip-gram size
ind = np.array(np.random.randint(0, int(target_word.shape[0]), size = 1000000), dtype = 'int32')
target_word = target_word[ind]
context_word = context_word[ind]
Y = Y[ind]

# SEARCH PARAMETERS FOR HYPERPARAMETER OPTIMIZATION
# https://medium.com/@crawftv/parameter-hyperparameter-tuning-with-bayesian-optimization-7acf42d348e1
print("INITIALIZE HYPERPARAMETER OPTIMIZATION")
dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform',
                         name='learning_rate')
dim_embedding_size = Integer(low=100, high=200, name='embedding_size')
#dim_num_dense_layers = Integer(low=1, high=5, name='num_dense_layers')
dim_activation = Categorical(categories=['sigmoid', 'softmax'],
                             name='activation_function')
dim_batch_size = Integer(low=32, high=256, name='batch_size')
dim_epochs = Integer(low=5, high=100, name='num_epochs')
dim_adam_decay = Real(low=1e-6,high=1e-2,name="adam_decay")

dimensions = [dim_learning_rate,
                dim_embedding_size,
                #dim_num_dense_layers,
                dim_activation,
                dim_batch_size,
                dim_epochs,
                dim_adam_decay]

# define default parameters
default_parameters = [1e-3, 100, 'sigmoid', 64, 5, 1e-3]

# MODEL CREATION
print("MODEL GENERATION")

# batch generator
# iterate systematically
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

# use random integers
# does not work properly at the moment bc random integers are always the same
def batch_generator2(target, context, Y, batch_size):
    counter = 0
    while True:
        idx = np.array(np.random.randint(0, target.shape[0], size = batch_size), dtype = 'int32')

        target_batch = target[idx]
        context_batch = context[idx]
        Y_batch = Y[idx]

        counter += 1

        #print([target_batch, context_batch], Y_batch)
        yield ([target_batch, context_batch], Y_batch)

        gc.collect()

# split data into training and testing (validation) data sets
target_train, target_test, context_train, context_test, Y_train, Y_test = train_test_split(target_word, context_word, Y, test_size=valSplit)

def create_model (learning_rate, embedding_size, activation_function, adam_decay, batch_size):
    input_target = keras.layers.Input(((1,)), name='target_word')
    input_context = keras.layers.Input(((1,)), name='context_word')

    # embed input layers
    # https://github.com/keras-team/keras/issues/3110
    embedding = Embedding(input_dim = vocab_size,
                            output_dim = embedding_size,
                            input_length = 1,
                            embeddings_initializer = 'glorot_uniform',
                            name = 'embedding')
    # later create lookup table with weights so that one could initialize the embedding layer with pretrained weights

    # apply embedding
    target = embedding(input_target)
    target = Reshape((embedding_size,1), name='target_embedding')(target) # every individual skip-gram has dimension embedding x 1
    context = embedding(input_context)
    context = Reshape((embedding_size,1), name='context_embedding')(context)


    dot_product = dot([target, context], axes = 1, normalize = True, name = 'dot_product')
    dot_product = Reshape((1,))(dot_product)

    # add the sigmoid dense layer
    output = Dense(1, activation = activation_function, kernel_initializer = 'glorot_uniform', name='1st_layer')(dot_product)
    output = Dense(1, activation = activation_function, kernel_initializer = 'glorot_uniform', name='2nd_layer')(output)

    # create the primary training model
    model = Model(inputs=[input_target, input_context], outputs=output)
    adam = Adam(lr=learning_rate, decay=adam_decay)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy']) # binary for binary decisions, categorical for classifications

    print('metrics: {}'.format(model.metrics_names))

    return model

@use_named_args(dimensions=dimensions)
def fitness(learning_rate, embedding_size, activation_function, batch_size, num_epochs, adam_decay):

    model = create_model(learning_rate = learning_rate,
                            embedding_size = embedding_size,
                            activation_function = activation_function,
                            adam_decay = adam_decay,
                            batch_size = batch_size)

    # apply batch generator
    train_generator = batch_generator(target_train, context_train, Y_train, batch_size)
    test_generator = batch_generator(target_test, context_test, Y_test, batch_size)

    steps = np.ceil((target_train.shape[0]/batchSize)*0.1)
    val_steps = np.ceil((target_test.shape[0]/batchSize)*0.1)

    #named blackbox becuase it represents the structure
    blackbox = model.fit_generator(generator=train_generator,
                        validation_data=test_generator,
                        steps_per_epoch = steps,
                        validation_steps = val_steps,
                        epochs=num_epochs,
                        verbose=2,
                        workers=workers,
                        use_multiprocessing=True,
                        shuffle=True)

    #return the validation accuracy for the last epoch.
    accuracy = blackbox.history['val_accuracy'][-1]

    # Print the classification accuracy.
    print()
    print("Accuracy: {0:.2%}".format(accuracy))
    print()

    del model

    K.clear_session()
    tf.reset_default_graph()

    # the optimizer aims for the lowest score, so we return our negative accuracy
    return -accuracy

# gaussian
gp_result = gp_minimize(func = fitness,
                            dimensions = dimensions,
                            n_calls = 12,
                            noise = 0.01,
                            n_jobs = -1,
                            kappa = 5,
                            x0= default_parameters)

K.clear_session()
tf.reset_default_graph()

# gradient boosted regression trees
#gbrt_result = gbrt_minimize(func=fitness,
#                            dimensions=dimensions,
#                            n_calls=20,
#                            n_jobs=-1,
#                            x0=default_parameters)
#K.clear_session()
#tf.reset_default_graph()

### OUTPUT ###
print("SAVING")
gp = sorted(zip(gp_result.func_vals, gp_result.x_iters))
print("result of gaussian process hyperparameter optimization")
print(gp)

#gbrt = sorted(zip(gbrt_result.func_vals, gbrt_result.x_iters))
#print("result of gradient boosted regression trees hyperparameter optimization")
#print(gbrt)

gp_file = pd.DataFrame(gp)
pd.DataFrame.to_csv(gp_file, 'optimization/results/gp_new_{}.csv'.format(size), header=False)

#gbrt_file = pd.DataFrame(gbrt)
#pd.DataFrame.to_csv(gbrt_file, 'optimization/results/gbrt_new_{}.csv'.format(size), header=False)

# save weights of embedding matrix
df = pd.DataFrame(weights)
pd.DataFrame.to_csv(df, 'results/embedded_proteome/weights_{}.csv'.format(size), header=False)
df.head()
