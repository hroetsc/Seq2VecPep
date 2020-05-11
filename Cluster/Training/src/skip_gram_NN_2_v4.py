#!/usr/bin/env python


### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  convert tokens to numerical vectors using a skip-gram neural network
# input:        word pairs (target and context word) generated in skip_gram_NN_1
# output:       embedded tokens (weights and their IDs)
# author:       HR

print("### PROTEOME EMBEDDING USING SKIP-GRAM NEURAL NETWORK - 2 ###")


import os
import sys
import json
import gc
import numpy as np
import pandas as pd

#import keras
#from keras import layers
from keras import backend as K

from sklearn.model_selection import train_test_split

import tensorflow as tf

# overwrite namespace!
from tensorflow import keras
from tensorflow.keras import layers

import horovod.tensorflow as hvd

# clean
gc.enable()

print("-------------------------------------------------------------------------")
print('INITIALIZE HOROVOD')
# initialize horovod
hvd.init()


print("-------------------------------------------------------------------------")
print("ENVIRONMENT VARIABLES")

jobname = str(os.environ["SLURM_JOB_NAME"])
print("JOBNAME: ")
print(jobname)

nodelist = os.environ["SLURM_JOB_NODELIST"]
print("NODELIST: ")
print(nodelist)

nodename = os.environ["SLURMD_NODENAME"]
print("NODENAME: ")
print(nodename)

num_nodes = int(os.getenv("SLURM_JOB_NUM_NODES"))
print("NUMBER OF NODES: ")
print(num_nodes)

print("NUM GPUs AVAILABLE: ")
print(len(tf.config.experimental.list_physical_devices('GPU')))

tasks_per_node = int(os.getenv("SLURM_NTASKS_PER_NODE"))
print("TASKS PER NODE: ")
print(tasks_per_node)


print("-------------------------------------------------------------------------")

print('GPU CONFIGURATION')

# pin GPUs (each GPU gets single process)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

print("-------------------------------------------------------------------------")

# =============================================================================
# # HYPERPARAMETERS
# =============================================================================
print("SET HYPERPARAMETERS")

# epoch ned to be adjusted by number of GPUs
epochs = 500
epochs = int(np.ceil(epochs/hvd.size()))
print('number of epochs, adjusted by number of GPUs: ', epochs)

valSplit = 0.1


NUM_WORKERS = num_nodes*tasks_per_node
PER_WORKER_BATCH_SIZE = 32

print('per worker batch size: ', PER_WORKER_BATCH_SIZE)


print("-------------------------------------------------------------------------")

# =============================================================================
# # GENERATOR FUNCTION
# =============================================================================
print('BATCH GENERATOR FUNCTION')

class BatchGenerator(keras.utils.Sequence):

     def __init__(self, target, context, Y, batch_size):
         self.target, self.context, self.Y = target, context, Y
         self.batch_size = batch_size
         self.indices = np.arange(self.target.shape[0])

     def __len__(self):
         return int(np.ceil(len(self.target) / float(self.batch_size)))

     def __getitem__(self, idx):

         batch_target = self.target[idx*self.batch_size : (idx + 1)*self.batch_size]
         batch_context = self.context[idx*self.batch_size : (idx + 1)*self.batch_size]
         batch_Y = self.Y[idx*self.batch_size : (idx + 1)*self.batch_size]

         return [batch_target, batch_context], batch_Y

     def on_epoch_end(self):
         np.random.shuffle(self.indices)



print("-------------------------------------------------------------------------")

# =============================================================================
# # MODEL
# =============================================================================

print('BUILD MODEL')

def build_and_compile_model():

    tf.print('input')

    input_target = keras.Input(((1,)), name='target_word')
    input_context = keras.Input(((1,)), name='context_word')

    tf.print('embedding')
    # embed input layers
    embedding = layers.Embedding(input_dim = vocab_size,
                            output_dim = embeddingDim,
                            input_length = 1,
                            embeddings_initializer = 'he_uniform',
                            name = 'embedding')

    # apply embedding
    target = embedding(input_target)
    target = layers.Reshape((embeddingDim,1), name='target_embedding')(target) # every individual skip-gram has dimension embedding x 1
    context = embedding(input_context)
    context = layers.Reshape((embeddingDim,1), name='context_embedding')(context)

    tf.print('dot product')
    # dot product similarity - normalize to get value between -1 and 1!
    dot_product = layers.dot([target, context], axes = 1, normalize = True, name = 'dot_product')
    dot_product = layers.Reshape((1,))(dot_product)


    tf.print('dense layers')
    # add dense layers
    x = layers.Dense(64, activation = 'tanh', kernel_initializer = 'he_uniform', name='1st_dense')(dot_product)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation = 'tanh', kernel_initializer = 'he_uniform', name='2nd_dense')(x)


    tf.print('concatenation')
    # create the primary training model
    model = keras.Model(inputs=[input_target, input_context], outputs=output)

    tf.print('compile model')
    # wrap optimizer
    opt = tf.optimizers.Adam(0.001 * hvd.size())
    opt = hvd.DistributedOptimizer(opt)

    model.compile(loss=keras.losses.SquaredHinge(),
                    optimizer = opt,
                    metrics=[tf.keras.metrics.Accuracy()],
                    experimental_run_tf_function=False)

    tf.print('return model')

    return model


# =============================================================================
# # INPUT
# =============================================================================

print("-------------------------------------------------------------------------")

print("LOAD DATA")

skip_grams = pd.read_csv(snakemake.input['skip_grams'], sep = " ", header = None)
ids = pd.read_csv(snakemake.input['ids'], header = None)
params = pd.read_csv('params.csv', header = 0)

# get vocabulary size
vocab_size = len(ids.index) + 1
print("vocabulary size (number of target word IDs + 1): {}".format(vocab_size))

output = snakemake.output['weights']


for i in range(len(output)):

    embeddingDim = int(params.iloc[i,0])

    print('-----------------------------------------------')
    print('-----------------------------------------------')
    print('EMBEDDING DIMENSIONS: ', embeddingDim)
    print('-----------------------------------------------')
    print('-----------------------------------------------')

    # split skip-grams into target, context and label np.array()
    target_word = np.array(skip_grams.iloc[:,0], dtype = 'int32')
    context_word = np.array(skip_grams.iloc[:,1], dtype = 'int32')
    Y = np.array(skip_grams.iloc[:,2], dtype = 'int32')


    # =============================================================================
    # for testing!!!
    #ind = np.random.randint(0, target_word.shape[0], 100000)
    #target_word = target_word[ind]
    #context_word = context_word[ind]
    #Y = Y[ind]
    # =============================================================================

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


    model = build_and_compile_model()

    # =============================================================================
    # # TRAINING
    # =============================================================================
    # view model summary
    print(model.summary())

    print("MODEL TRAINING")
    print('model metrics: {}'.format(model.metrics_names))

    # split data into training and validation
    print("split word pairs into training and validation data sets")
    target_train, target_test, context_train, context_test, Y_train, Y_test = train_test_split(target_word, context_word, Y, test_size=valSplit)

    # apply batch generator
    print("generating batches for model training")

    train_generator = BatchGenerator(target_train, context_train, Y_train, PER_WORKER_BATCH_SIZE)
    test_generator = BatchGenerator(target_test, context_test, Y_test, PER_WORKER_BATCH_SIZE)


    print('define callbacks')

    # early stopping if model is already converged
    es = tf.keras.callbacks.EarlyStopping(monitor = 'accuracy',
                                            mode = 'max',
                                            patience = 10,
                                            min_delta = 0.005,
                                            verbose = 1)

    # checkpoints
    callbacks = [
        es,
    ]

    # save chackpoints only on worker 0
    if hvd.rank() == 0:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath='/scratch2/hroetsc/Seq2Vec',
                                                    monitor = 'accuracy',
                                                    mode = 'max',
                                                    safe_best_only = True,
                                                    verbose = 1))


    # fit model
    print("fit the model")

    # one epoch: one complete iteration over dataset
    steps = np.ceil(target_train.shape[0]/PER_WORKER_BATCH_SIZE)
    val_steps = np.ceil(target_test.shape[0]/PER_WORKER_BATCH_SIZE)

    # adjust by number of GPUs
    steps = int(np.ceil(steps / hvd.size()))
    val_steps = int(np.ceil(val_steps / hvd.size()))

    fit = model.fit_generator(generator = train_generator,
                        validation_data = test_generator,
                        steps_per_epoch = steps,
                        validation_steps = val_steps,
                        validation_freq = 1,
                        epochs = epochs,
                        callbacks = callbacks,
                        initial_epoch = 0,
                        verbose = 2 if hvd.rank() == 0 else 0,
                        max_queue_size = 1,
                        shuffle = True)


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
    pd.DataFrame.to_csv(df, snakemake.output['weights'][i], header=False)
    df.head()

    # save model
    model.save(snakemake.output['model'][i])

    # save accuracy and loss
    m = open(snakemake.output['metrics'][i], 'w')
    m.write("accuracy \t {} \n val_accuracy \t {} \n loss \t {} \n val_loss \t {}".format(fit.history['accuracy'], fit.history['val_accuracy'], fit.history['loss'], fit.history['val_loss']))
    m.close()

    # clear session
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
