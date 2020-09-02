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

epochs = 800
batchSize = 16
pseudocounts = 1

epochs = int(np.ceil(epochs / hvd.size()))
batchSize = batchSize * hvd.size()

print('number of epochs, adjusted by number of GPUs: ', epochs)
print('batch size, adjusted by number of GPUs: ', batchSize)
print('number of pseudocounts: ', pseudocounts)
print('using scaled input data: False --> using AAindex as features')
print("-------------------------------------------------------------------------")

########## part 1: fit model ##########
### INPUT ###
print('LOAD DATA')

ext = ""
print('extension: ', ext)

subs = 'OPT'
print('subset: ', subs)

## on the cluster
tokensAndCounts_train = pd.read_csv(str('/scratch2/hroetsc/Hotspots/data/'+ext+'windowTokens_'+subs+'training.csv'))
emb_train = str('/scratch2/hroetsc/Hotspots/data/AAindex_'+ext+'embMatrices_'+subs+'training.dat')
acc_train = str('/scratch2/hroetsc/Hotspots/data/AAindex_'+ext+'embMatricesAcc_'+subs+'training.dat')


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

def KL_expansion(inp_data):
    val, vec = np.linalg.eig(np.cov(inp_data))
    klt = np.dot(vec, inp_data)
    return klt


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

            # normalize input data (z-transformation)
            # dt = (dt - np.mean(dt)) / np.std(dt)

            # make sure to pass 4D-Tensor to model: (batchSize, depth, height, width)
            dt = dt.reshape((tokPerWindow, embeddingDim))

            # KL transformation
            # dt = KL_expansion(dt)

            # for 2D convolution:
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

    embMatrix = np.array(embMatrix, dtype='float32')

    # output: reformatted tokens and counts, embedding matrix
    return tokens, counts, embMatrix


# apply
tokens, counts = format_input(tokensAndCounts_train)
tokens, counts, emb = open_and_format_matrices(tokens, counts, emb_train, acc_train)

print(counts)


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
def dense_layer(prev_layer, nodes):
    norm = layers.BatchNormalization(trainable=True)(prev_layer)
    dense = layers.Dense(nodes, activation='selu',
                         kernel_initializer=keras.initializers.LecunNormal())(norm)
    return dense


def convolution_2D(prev_layer, no_filters, kernel_size, strides, pooling=False):
    conv = layers.Conv2D(filters=no_filters,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding='same',
                         kernel_initializer=keras.initializers.HeNormal(),
                         data_format='channels_first')(prev_layer)

    norm = layers.BatchNormalization(trainable=True)(conv)
    act = layers.Activation('relu')(norm)

    if pooling:
        pool = layers.MaxPool2D(pool_size=2,
                                strides=2,
                                data_format='channels_first',
                                padding='same')(act)
        return pool

    else:
        return act


def custom_loss(y_true, y_pred):
    custom_loss = kb.sum(kb.square(y_true - y_pred))
    return custom_loss


# function that returns model
def build_and_compile_model():
    ## input
    tf.print('model input')
    # inp = keras.Input(shape=(tokPerWindow, embeddingDim),
    #                   name='input')
    inp = keras.Input(shape=(1, tokPerWindow, embeddingDim),
                      name='input')

    dense = dense_layer(inp, embeddingDim)
    dense = dense_layer(dense, embeddingDim)
    dense = dense_layer(dense, embeddingDim)
    dense = dense_layer(dense, embeddingDim)

    flat = layers.Flatten()(dense)

    dense = dense_layer(flat, 1024)
    dense = dense_layer(dense, 128)
    dense = dense_layer(dense, 64)
    dense = dense_layer(dense, 32)
    dense = dense_layer(dense, 16)

    out_norm = layers.BatchNormalization(trainable=True)(dense)
    out = layers.Dense(1, activation='linear',
                       kernel_initializer=keras.initializers.HeNormal(),
                       name='output')(out_norm)

    ## concatenate to model
    model = keras.Model(inputs=inp, outputs=out)

    ## compile model
    tf.print('compile model')

    lr = 0.001 * hvd.size()
    tf.print('learning rate, adjusted by number of GPUS: ', lr)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    opt = hvd.DistributedOptimizer(opt)

    model.compile(loss=custom_loss,
                  optimizer=opt,
                  metrics=['mean_squared_error', 'mean_absolute_error', 'accuracy', 'mean_absolute_percentage_error'],
                  experimental_run_tf_function=False)

    # for reproducibility during optimization
    tf.print('......................................................')
    tf.print('BUILD A MODEL FROM SCRATCH')
    tf.print('optimizer: Adam')
    tf.print('loss: custom --> squared error')
    tf.print('activation function dense layers: selu/lecun normal')
    tf.print('regularization: none')
    tf.print('kernel initialization: he normal')
    tf.print('bias initialization: he normal')
    tf.print('using batch normalization: yes')
    tf.print('using Dropout layer: no')
    tf.print('......................................................')

    return model


#### train model
print('MODEL TRAINING')
# define callbacks - adapt later for multi-node training
# early stopping if model is already converged
# early stopping does not work at the moment
# Error: Error occurred when finalizing GeneratorDataset iterator: Failed precondition: Python interpreter state is not initialized. The process may be terminated.
#         [[{{node PyFunc}}]]
# only append to callbacks on worker 0?

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                      mode='min',
                                      patience=2,
                                      min_delta=0.005,
                                      verbose=1,
                                      restore_best_weights=True)

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
                                                        save_weights_only=False))
    callbacks.append(es)

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
# print('using sequence generator')
fit = model.fit(x=emb_train,
                y=counts_train,
                batch_size=batchSize,
                validation_batch_size=batchSize,
                validation_data=(emb_val, counts_val),
                steps_per_epoch=steps,
                validation_steps=val_steps,
                epochs=epochs,
                callbacks=callbacks,
                initial_epoch=0,
                max_queue_size=256,
                verbose=2 if hvd.rank() == 0 else 0,
                shuffle=True)
# fit = model.fit(SequenceGenerator(emb_train, counts_train, batchSize),
#                 validation_data=SequenceGenerator(emb_val, counts_val, batchSize),
#                 steps_per_epoch=steps,
#                 validation_steps=val_steps,
#                 epochs=epochs,
#                 callbacks=callbacks,
#                 initial_epoch=0,
#                 max_queue_size=256,
#                 verbose=2 if hvd.rank() == 0 else 0,
#                 shuffle=True)

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
tokensAndCounts_test = pd.read_csv(str('/scratch2/hroetsc/Hotspots/data/'+ext+'windowTokens_'+subs+'testing.csv'))
emb_test = str('/scratch2/hroetsc/Hotspots/data/AAindex_'+ext+'embMatrices_'+subs+'testing.dat')
acc_test = str('/scratch2/hroetsc/Hotspots/data/AAindex_'+ext+'embMatricesAcc_'+subs+'testing.dat')

## on local machine
# tokensAndCounts_test = pd.read_csv('data/windowTokens_OPTtesting.csv')
# emb_test = 'data/embMatrices_testing.dat'
# acc_test = 'data/embMatricesAcc_testing.dat'


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
prediction = pd.DataFrame({"Accession": tokens_test[:, 0],
                           "window": tokens_test[:, 1],
                           "count": counts_test,
                           "pred_count": pred.flatten()})

### OUTPUT ###
print('SAVE PREDICTED COUNTS')

pd.DataFrame.to_csv(prediction,
                    '/scratch2/hroetsc/Hotspots/results/model_predictions_rank{}.csv'.format(int(hvd.rank())),
                    index=False)

tf.keras.backend.clear_session()
