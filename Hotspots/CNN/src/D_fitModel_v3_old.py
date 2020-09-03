### HEADER ###
# HOTSPOT PREDICTION
# description: set up neural network to predict hotspot density along the human proteome
# input: table with tokens in sliding windows and scores,
# output: model, metrics, predictions for test data set
# author: HR

# using DenseNet structure

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import horovod.tensorflow.keras as hvd

tf.keras.backend.clear_session()
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

epochs = 100
batchSize = 32
pseudocounts = 1

epochs = int(np.ceil(epochs / hvd.size()))
batchSize = batchSize * hvd.size()

print('number of epochs, adjusted by number of GPUs: ', epochs)
print('batch size, adjusted by number of GPUs: ', batchSize)
print('number of pseudocounts: ', pseudocounts)
print('using scaled input data: True')
print("-------------------------------------------------------------------------")

########## part 1: fit model ##########
### INPUT ###
print('LOAD DATA')

## on the cluster
tokensAndCounts_train = pd.read_csv('/scratch2/hroetsc/Hotspots/data/windowTokens_OPTtraining.csv')
emb_train = '/scratch2/hroetsc/Hotspots/data/embMatrices_training.dat'
acc_train = '/scratch2/hroetsc/Hotspots/data/embMatricesAcc_training.dat'

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
            # scale all values between 0 and 1
            dt = (dt - np.min(dt)) / (np.max(dt) - np.min(dt))

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
# composite function: batch normalization, relu, 3x3 convolution
# with bottleneck
def composite(inp_layer, filters):
    # bottleneck
    l = layers.BatchNormalization(trainable=True)(inp_layer)
    l = layers.Activation('relu')(l)
    l = layers.ZeroPadding2D((1,1))(l)
    l = layers.Conv2D(filters=filters,
                      kernel_size=(1, 1),
                      strides=1,
                      padding='same',
                      kernel_regularizer=keras.regularizers.l2(l=0.0001),
                      kernel_initializer=tf.keras.initializers.HeNormal(),
                      data_format='channels_first')(l)

    # actual composition
    l = layers.BatchNormalization(trainable=True)(l)
    l = layers.Activation('relu')(l)
    l = layers.ZeroPadding2D((1, 1))(l)
    l = layers.Conv2D(filters=filters,
                      kernel_size=(3,3),
                      strides=1,
                      padding='same',
                      kernel_regularizer=keras.regularizers.l2(l=0.0001),
                      kernel_initializer=tf.keras.initializers.HeNormal(),
                      data_format='channels_first')(l)
    return l

# layers between blocks: transition layers (do convolution and pooling)
# batch normalization layer and an 1×1 convolutional layer followed by a 2×2 average pooling layer
# compression to reduce the number of feature maps at transition
def transition_layers(inp_layer, theta):
    m = inp_layer.shape[1]
    n_maps = int(np.floor(m * theta))

    l = layers.BatchNormalization(trainable=True)(inp_layer)
    l = layers.Activation('relu')(l)

    l = layers.Conv2D(filters=n_maps,
                      kernel_size=(1,1),
                      strides=1,
                      padding='same',
                      kernel_regularizer=keras.regularizers.l2(l=0.0001),
                      kernel_initializer=tf.keras.initializers.HeNormal(),
                      use_bias=False,
                      data_format='channels_first')(inp_layer)

    l = layers.AveragePooling2D(pool_size=(2,2),
                                padding='same',
                                data_format='channels_first')(l)
    return l


# dense block
def dense_block(inp_layer, n_layers, n_filters, k):
    for i in range(n_layers):
        conv_out = composite(inp_layer, n_filters)
        inp_layer = layers.Concatenate(axis=2)([conv_out, inp_layer])

        n_filters += k

    return inp_layer, n_filters


# build dense selu layers with batch norm
def dense_layer(prev_layer, nodes):
    norm = layers.BatchNormalization(trainable=True)(prev_layer)
    dense = layers.Dense(nodes, activation='selu',
                         kernel_regularizer=keras.regularizers.l2(l=0.0001),
                         kernel_initializer=tf.keras.initializers.LecunNormal())(norm)
    return dense


# function that returns model
def build_and_compile_model():
    ## input
    tf.print('model input')
    inp = keras.Input(shape=(1, tokPerWindow, embeddingDim))

    ## hyperparameters
    num_filters = 32 # starting filter value
    num_blocks = 3
    num_layers_per_block = 80
    theta = 0.5  # compression factor
    k = 24  # growth rate

    # assemble all dense blocks and connect with transition layers
    tf.print('build DenseNet')
    # initial convolution
    t = layers.BatchNormalization(trainable=True)(inp)
    t = layers.Conv2D(filters=num_filters,
                      kernel_size=(3,3),
                      strides=1,
                      padding='same',
                      kernel_regularizer=keras.regularizers.l2(l=0.0001),
                      kernel_initializer=tf.keras.initializers.HeNormal(),
                      use_bias=False,
                      data_format='channels_first')(t)
    # dense blocks
    for i in range(num_blocks):
        t, num_filters = dense_block(t, num_layers_per_block, num_filters, k)
        t = transition_layers(t, num_filters, theta)

    t = layers.GlobalAveragePooling2D()

    out_norm = layers.BatchNormalization(trainable=True)(t)
    out = layers.Dense(1, activation='linear', # no negative predictions
                       kernel_regularizer=keras.regularizers.l2(l=0.0001),
                       kernel_initializer=tf.keras.initializers.HeNormal(),
                       name='output')(out_norm)


    ## concatenate to model
    model = keras.Model(inputs=inp, outputs=out)

    ## compile model
    tf.print('compile model')
    opt = tf.keras.optimizers.Adagrad(learning_rate=0.0001 * hvd.size())
    opt = hvd.DistributedOptimizer(opt)

    model.compile(loss=keras.losses.MeanSquaredError(),
                  optimizer=opt,
                  metrics=['accuracy', 'mean_absolute_error'],
                  experimental_run_tf_function=False)


    # for reproducibility during optimization
    tf.print('......................................................')
    tf.print('optimizer: Adagrad')
    tf.print("learning rate: 0.001")
    tf.print('number of dense blocks: ', num_blocks)
    tf.print('size of dense block: ', num_layers_per_block)
    tf.print('growth rate: ', k)
    tf.print('compression factor: ', theta)
    tf.print('channels: first')
    tf.print('activation function: relu/he_normal')
    tf.print('number of dense layers before output layer: 0')
    tf.print('starting filter value: ', num_filters)
    tf.print('output activation function: linear')
    tf.print('regularization: L2 (0001)')
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
tokensAndCounts_test = pd.read_csv('/scratch2/hroetsc/Hotspots/data/windowTokens_OPTtesting.csv')
emb_test = '/scratch2/hroetsc/Hotspots/data/embMatrices_testing.dat'
acc_test = '/scratch2/hroetsc/Hotspots/data/embMatricesAcc_testing.dat'

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
                     batch_size=4,
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
