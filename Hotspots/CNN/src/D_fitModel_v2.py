### HEADER ###
# HOTSPOT PREDICTION
# description: set up neural network to predict hotspot density along the human proteome
# input: table with tokens in sliding windows and scores,
# output: model, metrics, predictions for test data set
# author: HR

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

epochs = 1600
batchSize = 32

epochs = int(np.ceil(epochs / hvd.size()))
batchSize = batchSize * hvd.size()

print('number of epochs, adjusted by number of GPUs: ', epochs)
print('batch size, adjusted by number of GPUs: ', batchSize)

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

    # log-transform counts
    counts = np.log2((counts + 1))

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

            # z-transform array
            # dt = (dt - np.mean(dt)) / np.std(dt)

            embMatrix[b] = dt.reshape((tokPerWindow, embeddingDim))

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
        batch_emb = self.emb[idx * self.batchSize: (idx + 1) * self.batchSize, :, :]
        batch_counts = self.counts[idx * self.batchSize: (idx + 1) * self.batchSize]

        return (batch_emb, batch_counts)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


# split data into training and validation
emb_train, emb_val, counts_train, counts_val = train_test_split(emb, counts, test_size=.1)


### build and compile model
# build dense relu layers with batch norm and dropout
def dense_layer(prev_layer, nodes):
    norm = layers.BatchNormalization(trainable=True)(prev_layer)
    dense = layers.Dense(nodes, activation='relu',
                         kernel_regularizer=keras.regularizers.l2(l=0.01),
                         kernel_initializer=tf.keras.initializers.HeNormal())(norm)
    return dense


# activation and batch normalization
def bn_relu(inp_layer):
    bn = layers.BatchNormalization(trainable=True)(inp_layer)
    relu = layers.Activation('relu')(bn)
    return relu


# residual blocks (convolutions)
def residual_block(inp_layer, downsample, filters, kernel_size):
    y = layers.Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=(1 if not downsample else 2),
                      padding='same',
                      kernel_regularizer=keras.regularizers.l2(l=0.01),
                      kernel_initializer=tf.keras.initializers.HeNormal(),
                      data_format='channels_first')(inp_layer)
    y = bn_relu(y)
    y = layers.Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=1,
                      padding='same',
                      kernel_regularizer=keras.regularizers.l2(l=0.01),
                      kernel_initializer=tf.keras.initializers.HeNormal(),
                      data_format='channels_first')(y)

    if downsample:
        inp_layer = layers.Conv2D(filters=filters,
                                  kernel_size=1,
                                  strides=2,
                                  padding='same',
                                  kernel_regularizer=keras.regularizers.l2(l=0.01),
                                  kernel_initializer=tf.keras.initializers.HeNormal(),
                                  data_format='channels_first')(inp_layer)

    out = layers.Add()([inp_layer, y])
    out = bn_relu(out)

    return out


# function that returns model
def build_and_compile_model():
    ## input
    tf.print('model input')
    inp = keras.Input(shape=(tokPerWindow, embeddingDim, 1),
                      name='input')

    ## hyperparameters
    # starting filter value
    num_filters = 16
    kernel_size = 3
    # number and size of residual blocks
    num_blocks_list = [4, 4, 4]

    ## convolutional layers (ResNet)
    tf.print('residual blocks')

    # structure of residual blocks:
    # a) original: weight-BN-ReLU-weight-BN-addition-ReLU
    # b) BN after addition: weight-BN-ReLU-weight-addition-BN-ReLU --> currently used
    # c) ReLU before addition: weight-BN-ReLU-weight-BN-ReLU-addition
    # d) full pre-activation (SpliceAI): BN-ReLU-weight-BN-ReLU-weight-addition

    # initial convolution
    t = layers.BatchNormalization(trainable=True)(inp)
    t = layers.Conv2D(filters=num_filters,
                      kernel_size=kernel_size,
                      strides=1,
                      padding='same',
                      kernel_regularizer=keras.regularizers.l2(l=0.01),
                      kernel_initializer=tf.keras.initializers.HeNormal(),
                      data_format='channels_first')(t)
    t = bn_relu(t)

    # residual blocks
    for i in range(len(num_blocks_list)):
        no_blocks = num_blocks_list[i]
        for j in range(no_blocks):
            t = residual_block(t,
                               downsample=(j == 0 and i != 0),
                               filters=num_filters,
                               kernel_size=kernel_size)
        num_filters *= 2

    t = layers.AveragePooling2D(4)(t)
    flat = layers.Flatten()(t)

    ## dense layers
    tf.print('dense layers')
    # fully-connected layers with L2-regularization, batch normalization and dropout
    dense1 = dense_layer(flat, 1024)

    out_norm = layers.BatchNormalization(trainable=True)(dense1)
    out = layers.Dense(1, activation='relu', # no negative predictions
                       kernel_regularizer=keras.regularizers.l2(l=0.01),
                       kernel_initializer=tf.keras.initializers.HeNormal(),
                       name='output')(out_norm)


    ## concatenate to model
    model = keras.Model(inputs=inp, outputs=out)

    ## compile model
    tf.print('compile model')
    opt = tf.keras.optimizers.Adagrad(learning_rate=0.001 * hvd.size())
    opt = hvd.DistributedOptimizer(opt)

    model.compile(loss=keras.losses.MeanSquaredError(),
                  optimizer=opt,
                  metrics=['mean_absolute_error', 'mean_absolute_percentage_error', 'cosine_proximity'],
                  experimental_run_tf_function=False)

    # for reproducibility during optimization
    tf.print('optimizer: Adagrad')
    tf.print("learning rate: 0.001")
    tf.print('number/size of residual blocks: [4,4,4]')
    tf.print('number of dense layers: 2')
    tf.print('starting filter value: 16')
    tf.print('regularization: L2')
    tf.print('using batch normalization: yes')
    tf.print('using Dropout layer: no')

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
prediction = pd.DataFrame({"count": counts_test.flatten(),
                           "prediction": pred.flatten()})

### OUTPUT ###
print('SAVE PREDICTED COUNTS')

if hvd.rank() == 0:
    pd.DataFrame.to_csv(prediction, '/scratch2/hroetsc/Hotspots/results/model_predictions.csv', index=False)

tf.keras.backend.clear_session()
