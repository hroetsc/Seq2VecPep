### HEADER ###
# HOTSPOT PREDICTION
# description: set up neural network to predict hotspot density along the human proteome
# input: table with tokens in sliding windows and scores,
# output: model, metrics, predictions for test data set
# author: HR

# using simple conv / dense structure

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as kb

tf.keras.backend.clear_session()

import horovod.tensorflow.keras as hvd

hvd.init()

### initialize Horovod ###
print("-------------------------------------------------------------------------")
print("ENVIRONMENT VARIABLES")

jobname = str(os.environ["SLURM_JOB_NAME"])
print("JOBNAME: ", jobname)
nodelist = os.environ["SLURM_JOB_NODELIST"]
print("NODELIST: ", nodelist)
nodename = os.environ["SLURMD_NODENAME"]
print("NODENAME: ", nodename)
num_nodes = int(os.getenv("SLURM_JOB_NUM_NODES"))
print("NUMBER OF NODES: ", num_nodes)
print("NUM GPUs AVAILABLE: ", len(tf.config.experimental.list_physical_devices('GPU')))

print("-------------------------------------------------------------------------")

print('HOROVOD CONFIGURATION')

print('number of Horovod processes on current node: ', hvd.local_size())
print('rank of the current process: ', hvd.rank())  # node
print('local rank of the current process: ', hvd.local_rank())  # process on node
print('Is MPI multi-threading supported? ', hvd.mpi_threads_supported())
print('Is MPI enabled? ', hvd.mpi_enabled())
print('Is Horovod compiled with MPI? ', hvd.mpi_built())
print('Is Horovod compiled with NCCL? ', hvd.nccl_built())

print("-------------------------------------------------------------------------")

### initialize GPU training environment
print('GPU INITIALIZATION')
# pin GPUs (each GPU gets single process)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

### HYPERPARAMETERS ###
print('HYPERPARAMETERS')
embeddingDim = 128
tokPerWindow = 8

epochs = 400
batchSize = 16
pseudocounts = 1

epochs = int(np.ceil(epochs / hvd.size()))
batchSize = batchSize * hvd.size()

print('number of epochs, adjusted by number of GPUs: ', epochs)
print('batch size, adjusted by number of GPUs: ', batchSize)
print('number of pseudocounts: ', pseudocounts)
print('using scaled input data: False')
print("-------------------------------------------------------------------------")

########## part 1: fit model ##########
### INPUT ###
print('LOAD DATA')

## on the cluster
tokensAndCounts_train = pd.read_csv('/scratch2/hroetsc/Hotspots/data/windowTokens_OPTtraining.csv')
emb_train = '/scratch2/hroetsc/Hotspots/data/embMatrices_OPTtraining.dat'
acc_train = '/scratch2/hroetsc/Hotspots/data/embMatricesAcc_OPTtraining.dat'

tokensAndCounts_test = pd.read_csv('/scratch2/hroetsc/Hotspots/data/windowTokens_OPTtesting.csv')
emb_test = '/scratch2/hroetsc/Hotspots/data/embMatrices_OPTtesting.dat'
acc_test = '/scratch2/hroetsc/Hotspots/data/embMatricesAcc_OPTtesting.dat'

## on local machine
# tokensAndCounts_train = pd.read_csv('data/windowTokens_OPTtraining.csv')
# emb_train = 'data/embMatrices_training.dat'
# acc_train = 'data/embMatricesAcc_training.dat'

# tokensAndCounts_test = pd.read_csv('data/windowTokens_OPTtesting.csv')
# emb_test = 'data/embMatrices_testing.dat'
# acc_test = 'data/embMatricesAcc_testing.dat'


### MAIN PART ###
print('FORMAT INPUT AND GET EMBEDDING MATRICES')


### format input
def format_input(tokensAndCounts, return_acc=False):
    tokens = np.array(tokensAndCounts.loc[:, ['Accession', 'tokens']], dtype='object')
    counts = np.array(tokensAndCounts['counts'], dtype='float32')

    # log-transform counts (+ pseudocounts)
    counts = np.log2((counts + pseudocounts))

    # get binary labels
    labels = np.where(counts == 0, 0, 1)

    print('number of features: ', counts.shape[0])

    return tokens, labels, counts


### get embedding matrices and bring all features in correct (same) order
# 32 bit floats/integers --> 4 bytes
# 128 dimensions, 8 tokens per window --> 1024 elements * 4 bytes = 4096 bytes per feature


def open_and_format_matrices(tokens, labels, counts, emb_path, acc_path):
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

            # normalize input data (z-transformation)
            # dt = (dt - np.mean(dt)) / np.std(dt)

            # make sure to pass 4D-Tensor to model: (batchSize, depth, height, width)
            dt = dt.reshape((tokPerWindow, embeddingDim))

            # for 2D convolution --> 5d input:
            embMatrix[b] = np.expand_dims(dt, axis=0)
            # embMatrix[b] = dt

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
    labels = labels[accMatrix]

    embMatrix = np.array(embMatrix, dtype='float32')

    # output: reformatted tokens and counts, embedding matrix
    return tokens, labels, counts, embMatrix


# apply
tokens, labels, counts = format_input(tokensAndCounts_train)
tokens, labels, counts, emb = open_and_format_matrices(tokens, labels, counts, emb_train, acc_train)

tokens_test, labels_test, counts_test = format_input(tokensAndCounts_test)
tokens_test, labels_test, counts_test, emb_test = open_and_format_matrices(tokens_test, labels_test, counts_test,
                                                                           emb_test, acc_test)


### build and compile model
def dense_layer(prev_layer, nodes, drop_prob):
    drop = layers.Dropout(drop_prob)(prev_layer)
    norm = layers.BatchNormalization(trainable=True)(drop)
    dense = layers.Dense(nodes, activation='selu',
                         kernel_initializer=keras.initializers.LecunNormal(),
                         kernel_regularizer=keras.regularizers.l2(0.01))(norm)
    return dense


def convolution(prev_layer, no_filters, kernel_size, strides, drop_prob, pooling=True):
    conv = layers.Convolution2D(filters=no_filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding='same',
                                kernel_initializer=keras.initializers.HeNormal(),
                                kernel_regularizer=keras.regularizers.l2(0.01),
                                data_format='channels_first')(prev_layer)

    norm = layers.BatchNormalization(trainable=True)(conv)
    drop = layers.Dropout(drop_prob)(norm)
    act = layers.Activation('relu')(drop)

    if pooling:
        pool = layers.MaxPool2D(pool_size=2,
                                strides=2,
                                data_format='channels_first',
                                padding='same')(act)
        return pool

    else:
        return act


# function that returns model
def build_and_compile_model():
    drop_prob = 0.5
    tf.print('dropout probability: ', drop_prob)

    ## input
    tf.print('model input')

    inp = keras.Input(shape=(1, tokPerWindow, embeddingDim),
                      name='input')

    # conv = convolution(inp, no_filters=16, kernel_size=3, strides=2, drop_prob=0.1)
    # conv = convolution(conv, no_filters=32, kernel_size=3, strides=2, drop_prob=0.1)
    # conv = convolution(conv, no_filters=64, kernel_size=3, strides=2, drop_prob=0.1)

    dense = layers.Flatten()(inp)

    no_dense = 8
    tf.print('number of dense layers: ', no_dense)

    nodes = 2048
    for l in range(no_dense):
        dense = dense_layer(dense, nodes, drop_prob)
        nodes *= 0.5

    norm = layers.BatchNormalization(trainable=True)(dense)

    # classification task
    classification = layers.Dense(1, activation='sigmoid',
                                  kernel_initializer=keras.initializers.HeNormal(),
                                  kernel_regularizer=keras.regularizers.l2(0.01),
                                  name='classification')(norm)

    # regression task
    regression = layers.Dense(1, activation='linear',
                              kernel_initializer=keras.initializers.HeNormal(),
                              kernel_regularizer=keras.regularizers.l2(0.01),
                              name='regression')(norm)

    ## concatenate to model
    model = keras.Model(inputs=inp, outputs=[classification, regression])

    ## compile model
    tf.print('compile model')

    lr = 0.001 * hvd.size()
    tf.print('learning rate, adjusted by number of GPUS: ', lr)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    opt = hvd.DistributedOptimizer(opt)

    losses = {'classification': 'binary_crossentropy',
              'regression': 'mean_squared_error'}

    loss_weights = {'classification': 1.0,
                    'regression': 1.0}
    tf.print(loss_weights.keys())
    tf.print(loss_weights.values())

    metrics = {'classification': ['binary_accuracy'],
               'regression': ['mean_absolute_error', 'mean_absolute_percentage_error', 'cosine_proximity']}

    model.compile(loss=losses,
                  loss_weights=loss_weights,
                  optimizer=opt,
                  metrics=metrics,
                  experimental_run_tf_function=False)

    # for reproducibility during optimization
    tf.print('......................................................')
    tf.print('BINARY CLASSIFICATION AND REGRESSION AT THE SAME TIME')
    tf.print('optimizer: Adam')
    tf.print('loss: binary_crossentropy and mean_squared_error')
    tf.print('activation function dense layers: selu/lecun normal')
    tf.print('regularization: L2 (0.01)')
    tf.print('kernel initialization: he normal')
    tf.print('using batch normalization: yes')
    tf.print('using Dropout layer: yes, ', drop_prob)
    tf.print('......................................................')

    return model


#### train model
print('MODEL TRAINING')
# define callbacks - adapt later for multi-node training
callbacks = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),  # broadcast initial variables from rank 0 to all other servers
]

# save chackpoints only on worker 0
if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath='/scratch2/hroetsc/Hotspots/results/model/ckpts',
                                                        monitor='val_loss',
                                                        mode='min',
                                                        safe_best_only=True,
                                                        verbose=1,
                                                        save_weights_only=True))

# define number of steps - make sure that no. of steps is the same for all ranks!
# otherwise, stalled ranks problem might occur
steps = int(np.ceil(counts.shape[0] / batchSize))
val_steps = int(np.ceil(counts_test.shape[0] / batchSize))

# adjust by number of GPUs
steps = int(np.ceil(steps / hvd.size()))
val_steps = int(np.ceil(val_steps / hvd.size()))

## fit model
model = build_and_compile_model()
model.summary()

print('train for {}, validate for {} steps per epoch'.format(steps, val_steps))
# print('using sequence generator')
fit = model.fit(x=emb,
                y=[labels, counts],
                batch_size=batchSize,
                validation_batch_size=batchSize,
                validation_data=(emb_test, [labels_test, counts_test]),
                steps_per_epoch=steps,
                validation_steps=val_steps,
                epochs=epochs,
                callbacks=callbacks,
                initial_epoch=0,
                max_queue_size=256,
                verbose=2 if hvd.rank() == 0 else 0,
                shuffle=True)

### OUTPUT ###
print('SAVE MODEL AND METRICS')

if hvd.rank() == 0:
    # save weights
    model.save_weights('/scratch2/hroetsc/Hotspots/results/model/weights.h5')
    # save entire model
    model.save('/scratch2/hroetsc/Hotspots/results/model/model.h5')

    # save metrics
    val = []
    name = list(fit.history.keys())
    for i, elem in enumerate(fit.history.keys()):
        val.append(fit.history[elem])

    m = list(zip(name, val))
    m = pd.DataFrame(m)
    pd.DataFrame.to_csv(m, '/scratch2/hroetsc/Hotspots/results/model_metrics.txt', header=False, index=False)

########## part 2: make prediction ##########
print('MAKE PREDICTION')
### MAIN PART ###
# make prediction
pred = model.predict(x=emb_test,
                     batch_size=batchSize,
                     verbose=1 if hvd.rank() == 0 else 0,
                     max_queue_size=256)

print('counts:')
print(counts_test)

print('prediction:')
print(pred)

# merge actual and predicted counts
prediction = pd.DataFrame({"Accession": tokens_test[:, 0],
                           "window": tokens_test[:, 1],
                           "label": labels_test,
                           "count": counts_test,
                           "pred_label": pred[0].flatten(),
                           "pred_count": pred[1].flatten()})

### OUTPUT ###
print('SAVE PREDICTED COUNTS')

pd.DataFrame.to_csv(prediction,
                    '/scratch2/hroetsc/Hotspots/results/model_predictions_rank{}.csv'.format(int(hvd.rank())),
                    index=False)

tf.keras.backend.clear_session()
