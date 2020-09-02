### HEADER ###
# HOTSPOT PREDICTION
# description: set up neural network to predict hotspot density along the human proteome
# input: table with tokens in sliding windows and scores,
# output: model, metrics, predictions for test data set
# author: HR

# using custom structure with splitted input

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

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

epochs = 800
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
# tokensAndCounts_train = pd.read_csv('/scratch2/hroetsc/Hotspots/data/windowTokens_training.csv')
# emb_train = '/scratch2/hroetsc/Hotspots/data/embMatrices_training.dat'
# acc_train = '/scratch2/hroetsc/Hotspots/data/embMatricesAcc_training.dat'

tokensAndCounts_train = pd.read_csv('/scratch2/hroetsc/Hotspots/data/windowTokens_OPTtraining.csv')
emb_train = '/scratch2/hroetsc/Hotspots/data/embMatrices_OPTtraining.dat'
acc_train = '/scratch2/hroetsc/Hotspots/data/embMatricesAcc_OPTtraining.dat'

## on local machine
# tokensAndCounts_train = pd.read_csv('data/windowTokens_OPTtraining.csv')
# emb_train = 'data/embMatrices_training.dat'
# acc_train = 'data/embMatricesAcc_training.dat'


### MAIN PART ###
print('FORMAT INPUT AND GET EMBEDDING MATRICES')


### format input
def format_input(tokensAndCounts):
    tokens = np.array(tokensAndCounts.loc[:, ['Accession', 'tokens']], dtype='object')
    counts = np.array(tokensAndCounts['counts'], dtype='float32')

    # log-transform counts (+ pseudocounts)
    counts = np.log2((counts + pseudocounts))

    print('number of features: ', counts.shape[0])
    return tokens, counts


### get embedding matrices and bring all features in correct (same) order
# 32 bit floats/integers --> 4 bytes
# 128 dimensions, 8 tokens per window --> 1024 elements * 4 bytes = 4096 bytes per feature

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


# apply
tokens, counts = format_input(tokensAndCounts_train)
tokens, counts, emb = open_and_format_matrices(tokens, counts, emb_train, acc_train)

#### batch generator
print('SEQUENCE GENERATOR')


# generator function
class SequenceGenerator(keras.utils.Sequence):

    def __init__(self, emb, counts, batchSize):
        self.emb, self.counts = emb, counts
        self.batchSize = batchSize

        self.indices = np.arange(self.counts.shape[0])

    def __len__(self):
        return int(np.ceil(len(self.counts) / float(self.batchSize)))

    def __getitem__(self, idx):
        batch_emb = self.emb[idx * self.batchSize: (idx + 1) * self.batchSize, :, :, :]
        batch_counts = self.counts[idx * self.batchSize: (idx + 1) * self.batchSize]

        # print(batch_emb.shape)

        return (batch_emb, batch_counts)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


# split data into training and validation
emb_train, emb_val, counts_train, counts_val = train_test_split(emb, counts, test_size=.1)


### build and compile model
# build dense relu layers with batch norm and dropout
def dense_layer(prev_layer, nodes):
    norm = layers.BatchNormalization(trainable=True)(prev_layer)
    dense = layers.Dense(nodes, activation='selu',
                         kernel_initializer=tf.keras.initializers.LecunNormal())(norm)
    return dense


# residual blocks (convolutions)
def residual_block(prev_layer, no_filters, kernel_size, strides):
    conv = layers.Conv2D(filters=no_filters,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding='same',
                         kernel_initializer=tf.keras.initializers.HeNormal(),
                         data_format='channels_first')(prev_layer)

    norm = layers.BatchNormalization(trainable=True)(conv)
    act = layers.LeakyReLU()(norm)

    pool = layers.MaxPool2D(pool_size=2,
                            strides=2,
                            data_format='channels_first',
                            padding='same')(act)
    return pool


# function that returns model
def build_and_compile_model():
    ## hyperparameters
    num_filters = 16  # starting filter value
    kernel_size = 3

    ## input
    tf.print('model input')
    inp = keras.Input(shape=(1, tokPerWindow, embeddingDim),
                      name='input')

    # split input into single tokens and perform dimension reduction on them
    token_outputs = [None] * tokPerWindow
    for t in range(tokPerWindow):
        # slice input tensor
        intermediate = inp[:, :, t, :]
        # add dense layers
        intermediate = layers.Dense(64,
                                    activation='selu',
                                    kernel_initializer=tf.keras.initializers.LecunNormal())(intermediate)
        # intermediate = layers.BatchNormalization(trainable=True)(intermediate)
        intermediate = layers.Dense(8,
                                    activation='selu',
                                    kernel_initializer=tf.keras.initializers.LecunNormal())(intermediate)

        token_outputs[t] = intermediate

    # concatenate reduced token embeddings --> shape should be (None,8,8)
    redIn = layers.Concatenate(axis=1, name='reduced_input')(token_outputs)
    redIn = layers.Reshape((tokPerWindow, tokPerWindow))(redIn)

    # expand dimensions --> should be (None, 1, 8, 8)
    redIn = tf.expand_dims(redIn, axis=1)

    ## convolutional layers
    tf.print('convolutional layers')
    # convolution on input token embeddings --> 256-d
    t_conv = residual_block(redIn, num_filters, kernel_size, strides=1)
    t_flat = layers.Flatten(name='flattened_redInp')(t_conv)

    # convolution on full input tensor --> 256-d
    # f_conv = residual_block(inp, num_filters, kernel_size, strides=2)
    # f_conv = residual_block(f_conv, num_filters * 2, kernel_size, strides=2)
    # f_flat = layers.Flatten(name='flattened_fullInp')(f_conv)

    # add the two flattened layers
    # all_flat = layers.Add(name='flattened_allInps')([t_flat, f_flat])

    ## dense layers
    tf.print('dense layers')
    # fully-connected layers with batch normalization

    dense1 = dense_layer(t_flat, 256)
    dense2 = dense_layer(dense1, 128)
    dense3 = dense_layer(dense2, 64)

    out_norm = layers.BatchNormalization(trainable=True)(dense3)
    out = layers.Dense(1, activation='linear',
                       kernel_initializer=tf.keras.initializers.HeNormal(),
                       name='output')(out_norm)

    ## concatenate to model
    model = keras.Model(inputs=inp, outputs=out)

    ## compile model
    tf.print('compile model')
    opt = tf.keras.optimizers.Adagrad(learning_rate=0.01 * hvd.size())
    opt = hvd.DistributedOptimizer(opt)

    model.compile(loss='mean_squared_logarithmic_error',
                  optimizer=opt,
                  metrics=['mean_squared_error', 'mean_absolute_percentage_error', 'mean_absolute_error'],
                  experimental_run_tf_function=False)

    # for reproducibility during optimization
    tf.print('......................................................')
    tf.print('CUSTOM STRUCTURE WITH SPLITTED INPUT ')
    tf.print('optimizer: Adagrad')
    tf.print("learning rate: 0.01")
    tf.print('loss: mean squared logarithmic error')
    tf.print('channels: first')
    tf.print('pooling: MaxPool2D, strides = 2')
    tf.print('activation function: leaky relu / he_normal')
    tf.print('number of dense layers before output layer: 3 (256-64, selu)')
    tf.print('output activation function: linear')
    tf.print('starting filter value: ', num_filters)
    tf.print('kernel size: ', kernel_size)
    tf.print('strides: depending on the desired output dimension (256)')
    tf.print('number of convolutions: 1 for token embeddings, 2 for full input tensor')
    tf.print('regularization: none')
    tf.print('using batch normalization: yes')
    tf.print('using Dropout layer: no')
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
steps = int(np.ceil(counts_train.shape[0] / batchSize))
val_steps = int(np.ceil(counts_val.shape[0] / batchSize))

# adjust by number of GPUs
steps = int(np.ceil(steps / hvd.size()))
val_steps = int(np.ceil(val_steps / hvd.size()))

## fit model
model = build_and_compile_model()
model.summary()

print('train for {}, validate for {} steps per epoch'.format(steps, val_steps))

fit = model.fit(SequenceGenerator(emb_train, counts_train, batchSize),
                validation_data=SequenceGenerator(emb_val, counts_val, batchSize),
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

### INPUT ###
## on the cluster
# tokensAndCounts_test = pd.read_csv('/scratch2/hroetsc/Hotspots/data/windowTokens_testing.csv')
# emb_test = '/scratch2/hroetsc/Hotspots/data/embMatrices_testing.dat'
# acc_test = '/scratch2/hroetsc/Hotspots/data/embMatricesAcc_testing.dat'

tokensAndCounts_test = pd.read_csv('/scratch2/hroetsc/Hotspots/data/windowTokens_OPTtesting.csv')
emb_test = '/scratch2/hroetsc/Hotspots/data/embMatrices_OPTtesting.dat'
acc_test = '/scratch2/hroetsc/Hotspots/data/embMatricesAcc_OPTtesting.dat'

## on local machine
# tokensAndCounts_test = pd.read_csv('data/windowTokens_OPTtesting.csv')
# emb_test = 'data/embMatrices_testing.dat'
# acc_test = 'data/embMatricesAcc_testing.dat'

## tmp!!!
# tokensAndCounts_test = pd.read_csv('data/windowTokens_benchmark.csv')
# emb_test = 'data/embMatrices_benchmark.dat'
# acc_test = 'data/embMatricesAcc_benchmark.dat'


### MAIN PART ###
# format testing data
tokens_test, counts_test = format_input(tokensAndCounts_test)
# get embedding matrices for testing data
tokens_test, counts_test, emb_test = open_and_format_matrices(tokens_test, counts_test, emb_test, acc_test)

# make prediction
pred = model.predict(x=emb_test,
                     batch_size=batchSize,
                     verbose=1 if hvd.rank() == 0 else 0,
                     max_queue_size=256)

print(pred)

# merge actual and predicted counts
prediction = pd.DataFrame({"tokens": tokens_test[:, 1],
                           "count": counts_test,
                           "prediction": pred.flatten()})

### OUTPUT ###
print('SAVE PREDICTED COUNTS')

pd.DataFrame.to_csv(prediction,
                    '/scratch2/hroetsc/Hotspots/results/model_predictions_rank{}.csv'.format(int(hvd.rank())),
                    index=False)

tf.keras.backend.clear_session()
