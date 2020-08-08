#!/usr/bin/env python

### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  convert tokens to numerical vectors using a skip-gram neural network
# input:        word pairs (target and context word), IDs generated in skip_gram_NN_1
# output:       embedded tokens (weights and their IDs)
# author:       HR

print("### PROTEOME EMBEDDING USING SKIP-GRAM NEURAL NETWORK - 2 ###")

import os
import gc
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

tf.keras.backend.clear_session()


import horovod.tensorflow.keras as hvd

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


print("-------------------------------------------------------------------------")

print('HOROVOD CONFIGURATION')

print('number of Horovod processes on current node: ', hvd.local_size())
print('rank of the current process: ', hvd.rank()) # node
print('local rank of the current process: ', hvd.local_rank()) # process on node
print('Is MPI multi-threading supported? ', hvd.mpi_threads_supported())
print('Is MPI enabled? ', hvd.mpi_enabled())
print('Is Horovod compiled with MPI? ', hvd.mpi_built())
print('Is Horovod compiled with NCCL? ', hvd.nccl_built())

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

# epochs need to be adjusted by number of GPUs

# early stopping does not work at the moment
# Error: Error occurred when finalizing GeneratorDataset iterator: Failed precondition: Python interpreter state is not initialized. The process may be terminated.
#         [[{{node PyFunc}}]]


epochs = 300
epochs = int(np.ceil(epochs/hvd.size()))
print('number of epochs, adjusted by number of GPUs: ', epochs)

valSplit = 0.2

NUM_WORKERS = num_nodes
BATCH_SIZE = 16*hvd.size()

print('per worker batch size: ', BATCH_SIZE)

### output - this is not ideal ...
checkpoints = '/scratch2/hroetsc/Seq2Vec/results/GENCODEml_model_w5_d100/ckpts'
weights_path = '/scratch2/hroetsc/Seq2Vec/results/GENCODEml_model_w5_d100/weights.h5'
model_path = '/scratch2/hroetsc/Seq2Vec/results/GENCODEml_model_w5_d100/model'
embedding_layer = '/scratch2/hroetsc/Seq2Vec/results/GENCODEml_w5_d100_embedding.csv'


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

    input_target = keras.Input(shape = (1,),
                                name='target_word')
    input_context = keras.Input(shape = (1,),
                                name='context_word')

    tf.print('embedding')
    # embed input layers
    embedding = layers.Embedding(input_dim = vocab_size,
                            output_dim = embeddingDim,
                            input_length = 1,
                            embeddings_initializer = 'he_uniform',
                            name = 'embedding')

    # apply embedding
    target = embedding(input_target)
    context = embedding(input_context)

    tf.print('dot product')
    # dot product similarity - normalize to get value between -1 and 1! - ??
    dot_product = layers.dot([target, context], axes = 2, normalize = False, name = 'dot_product')
    dot_product = layers.Flatten()(dot_product)

    # batch normalise before passing to dense layer
    norm = layers.BatchNormalization(trainable = True)(dot_product)

    tf.print('dense layer with batch normalisation')

    output = layers.Dense(1, activation = 'sigmoid', kernel_initializer = 'he_uniform', name='dense')(norm)


    tf.print('concatenation')
    # create the primary training model
    model = keras.Model(inputs=[input_target, input_context], outputs=output)


    tf.print('compile model')
    # standard optimizer: Adam
    opt = keras.optimizers.Adam(learning_rate= 0.001 * 2 * hvd.size(),
                                name = 'Adam')

    # wrap it into stochastic weight averaging framework
    # does not work with Horovod's DistributedOptimizer

    #opt = tfa.optimizers.SWA(opt,
    #                        start_averaging = 0,
    #                        average_period= int(np.ceil(epochs/10)),
    #                        name = 'SWA')

    #opt = tfa.optimizers.Triangular2CyclicalLearningRate(initial_learning_rate = 0.001 * 2 * hvd.size(),
    #                                                maximal_learning_rate = 0.1,
    #                                                step_size = int(np.ceil(epochs/20)),
    #                                                scale_mode = 'cycle',
    #                                                name = 'CyclicalLearningRate')

    # wrap it into Horovod framework
    opt = hvd.DistributedOptimizer(opt, name = 'DistributedOpt')

    model.compile(loss=keras.losses.BinaryCrossentropy(),
                    optimizer = opt,
                    metrics=['squared_hinge', 'categorical_hinge', 'mean_squared_error','accuracy'],
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

# get vocabulary size
vocab_size = len(ids.index) + 1
print("vocabulary size (number of target word IDs + 1): {}".format(vocab_size))

embeddingDim = 128


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
# shuffle skip-gram order
print('randomly shuffle skip-grams')
ind = np.random.randint(0, target_word.shape[0], target_word.shape[0])

# tmp !!!
#ind = np.random.randint(0, target_word.shape[0], 500000)

target_word = target_word[ind]
context_word = context_word[ind]
Y = Y[ind]
# =============================================================================

print('target word vector')
target_word = target_word.reshape(target_word.shape[0],)
print(target_word)

print('context word vector')
context_word = context_word.reshape(context_word.shape[0],)
print(context_word)

print('label vector')
Y = Y.reshape(Y.shape[0],)
# replace 0 by -1
# Y = np.where(Y == 0, -1, Y)
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

train_generator = BatchGenerator(target_train, context_train, Y_train, BATCH_SIZE)
test_generator = BatchGenerator(target_test, context_test, Y_test, BATCH_SIZE)


print('define callbacks')

# early stopping if model is already converged
es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                        mode = 'min',
                                        patience = epochs, # causes an error currently so don't let early stopping happen
                                        min_delta = 0.0005,
                                        verbose = 1)

# checkpoints
callbacks = [
    es,
    hvd.callbacks.BroadcastGlobalVariablesCallback(0), # broadcast initial variables from rank 0 to all other servers
]

# save chackpoints only on worker 0
if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath = checkpoints,
                                                monitor = 'val_loss',
                                                mode = 'min',
                                                safe_best_only = False,
                                                verbose = 1,
                                                save_weights_only=True))


# fit model
print("fit the model")

# one epoch: one complete iteration over dataset
steps = int(np.ceil(target_train.shape[0]/BATCH_SIZE))
val_steps = int(np.ceil(target_test.shape[0]/BATCH_SIZE))


# adjust by number of GPUs
steps = int(np.ceil(steps / hvd.size()))
val_steps = int(np.ceil(val_steps / hvd.size()))

print('train for {} steps, validate for {} steps per epoch'.format(steps, val_steps))

fit = model.fit(x = [target_word, context_word],
                    y = Y,
                    validation_split = valSplit,
                    batch_size = BATCH_SIZE,
                    validation_batch_size = BATCH_SIZE,
                    steps_per_epoch = steps,
                    validation_steps = val_steps,
                    validation_freq = 1,
                    epochs = epochs,
                    callbacks = callbacks,
                    initial_epoch = 0,
                    #verbose = 2 if hvd.rank() == 0 else 0,
                    verbose = 2,
                    shuffle = True)

#fit = model.fit_generator(generator = train_generator,
#                    validation_data = test_generator,
#                    steps_per_epoch = steps,
#                    validation_steps = val_steps,
#                    validation_freq = 1,
#                    epochs = epochs,
#                    callbacks = callbacks,
#                    initial_epoch = 0,
#                    #verbose = 2 if hvd.rank() == 0 else 0,
#                    verbose = 2,
#                    max_queue_size = 1,
#                    shuffle = True)


# =============================================================================
# ### OUTPUT ###
# =============================================================================

print("OUTPUT")

if hvd.rank() == 0:
    # save weights
    model.save_weights(weights_path)

    #save entire model
    model.save(model_path)

    # save weights of embedding matrix
    weights = model.layers[2].get_weights()[0] # weights of the embedding layer of target word
    df = pd.DataFrame(weights)
    pd.DataFrame.to_csv(df, embedding_layer, header=False)


    # save metrics
    val = []
    name = list(fit.history.keys())
    for i, elem in enumerate(fit.history.keys()):
        val.append(fit.history[elem])

    m = list(zip(name, val))
    m = pd.DataFrame(m)
    pd.DataFrame.to_csv(m, snakemake.output['metrics'], header=False, index = False)


# clear session
tf.keras.backend.clear_session()
