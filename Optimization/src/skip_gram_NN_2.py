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

from keras.utils import Sequence

from sklearn.model_selection import train_test_split

# import scripts for hyperparameter optimization
import skopt
from skopt import gbrt_minimize, gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer
from keras.optimizers import Adam

import matplotlib
import matplotlib.pyplot as plt
from skopt.plots import plot_convergence, plot_objective

gc.enable()


# HYPERPARAMETERS
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

# SEARCH PARAMETERS FOR HYPERPARAMETER OPTIMIZATION
# https://medium.com/@crawftv/parameter-hyperparameter-tuning-with-bayesian-optimization-7acf42d348e1
print("INITIALIZE HYPERPARAMETER OPTIMIZATION")
dim_embedding_size = Integer(low=5, high=500, name='embedding_size')
dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform',
                         name='learning_rate')
dim_adam_decay = Real(low=1e-6,high=1e-2,name="adam_decay")
dim_relu_units = Integer(low=16, high=512, name='relu_units')

dimensions = [dim_embedding_size,
                dim_learning_rate,
                dim_adam_decay,
                dim_relu_units]

# define default parameters
default_parameters = [100, 1e-3, 1e-3, 64]

# MODEL CREATION
print("MODEL GENERATION")

class BatchGenerator(keras.utils.Sequence):

     def __init__(self, target, context, Y, batch_size):
         self.target, self.context, self.Y = target, context, Y
         self.batch_size = batch_size

     def __len__(self):
         return int(np.ceil(len(self.target) / float(self.batch_size)))

     def __getitem__(self, idx):

         #print(idx)

         batch_target = self.target[idx*self.batch_size : (idx + 1)*self.batch_size]
         batch_context = self.context[idx*self.batch_size : (idx + 1)*self.batch_size]
         batch_Y = self.Y[idx*self.batch_size : (idx + 1)*self.batch_size]

         return [batch_target, batch_context], batch_Y

     def on_epoch_end(self):
         pass

# split data into training and testing (validation) data sets
target_train, target_test, context_train, context_test, Y_train, Y_test = train_test_split(target_word, context_word, Y, test_size=valSplit)

def create_model (embedding_size, learning_rate, adam_decay, relu_units):
    input_target = keras.layers.Input(((1,)), name='target_word')
    input_context = keras.layers.Input(((1,)), name='context_word')

    # embed input layers
    # https://github.com/keras-team/keras/issues/3110
    embedding = Embedding(input_dim = vocab_size,
                            output_dim = embedding_size,
                            input_length = 1,
                            embeddings_initializer = 'he_uniform',
                            name = 'embedding')
    # later create lookup table with weights so that one could initialize the embedding layer with pretrained weights

    # apply embedding
    target = embedding(input_target)
    target = Reshape((embedding_size,1), name='target_embedding')(target) # every individual skip-gram has dimension embedding x 1
    context = embedding(input_context)
    context = Reshape((embedding_size,1), name='context_embedding')(context)

    # dot product similarity - normalize to get value between 0 and 1!
    dot_product = dot([target, context], axes = 1, normalize = True, name = 'dot_product')
    dot_product = Reshape((1,))(dot_product)

    # add dense layer
    output = Dense(int(relu_units), activation = 'tanh', kernel_initializer = 'he_uniform', name='1st_dense')(dot_product)
    output = Dropout(0.5)(output)
    output = Dense(2, activation = 'tanh', kernel_initializer = 'he_uniform', name='2nd_dense')(output)

    # create the primary training model
    model = Model(inputs=[input_target, input_context], outputs=output)

    adam = Adam(lr=learning_rate, decay=adam_decay)
    model.compile(loss='squared_hinge', optimizer=adam, metrics=['accuracy']) # binary for binary decisions, categorical for classifications

    return model

@use_named_args(dimensions=dimensions)
def fitness(embedding_size, learning_rate, adam_decay, relu_units):

    model = create_model(embedding_size = embedding_size,
                        learning_rate = learning_rate,
                        adam_decay = adam_decay,
                        relu_units = relu_units)

    # apply batch generator
    train_generator = BatchGenerator(target_train, context_train, Y_train, batchSize)
    test_generator = BatchGenerator(target_test, context_test, Y_test, batchSize)

    #named blackbox becuase it represents the structure
    blackbox = model.fit_generator(generator = train_generator,
                        validation_data = test_generator,
                        epochs = epochs,
                        initial_epoch = 0,
                        verbose=2,
                        max_queue_size = 1,
                        workers = workers,
                        use_multiprocessing = False,
                        shuffle = False)

    #return the validation accuracy for the last epoch.
    accuracy = blackbox.history['val_accuracy'][-1]

    # Print the classification accuracy.
    print()
    print('embedding size: {} - learning rate: {} - adam decay: {} - input units: {}'.format(embedding_size, learning_rate, adam_decay, relu_units))
    print()
    print("Accuracy: {0:.2%}".format(accuracy))
    print()

    del model

    K.clear_session()
    tf.compat.v1.reset_default_graph()

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
tf.compat.v1.reset_default_graph()

# gradient boosted regression trees
gbrt_result = gbrt_minimize(func = fitness,
                            dimensions=dimensions,
                            n_calls = 12,
                            n_jobs = -1,
                            x0 = default_parameters)
K.clear_session()
tf.compat.v1.reset_default_graph()

print("PLOT GENERATION")
# convergence plot
gp_conv = plot_convergence(gp_result)
plt.savefig(snakemake.output['gp_conv'], dpi = 300)

gbrt_conv = plot_convergence(gbrt_result)
plt.savefig(snakemake.output['gbrt_conv'], dpi = 300)

# partial dependence plot
gp_obj = plot_objective(gp_result, n_points = 10)
plt.savefig(snakemake.output['gp_obj'], dpi = 300)

gbrt_obj = plot_objective(gbrt_result, n_points = 10)
plt.savefig(snakemake.output['gbrt_obj'], dpi = 300)

### OUTPUT ###
print("SAVE RESULTS")
gp = sorted(zip(gp_result.func_vals, gp_result.x_iters))
print("result of gaussian process hyperparameter optimization")
print(gp)

gbrt = sorted(zip(gbrt_result.func_vals, gbrt_result.x_iters))
print("result of gradient boosted regression trees hyperparameter optimization")
print(gbrt)

gp_file = pd.DataFrame(gp)
pd.DataFrame.to_csv(gp_file, snakemake.output['gp'], header=False, index = False)

gbrt_file = pd.DataFrame(gbrt)
pd.DataFrame.to_csv(gbrt_file, snakemake.output['gbrt'], header=False, index = False)
