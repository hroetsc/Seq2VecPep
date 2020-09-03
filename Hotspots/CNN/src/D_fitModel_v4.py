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



# generator function
class SequenceGenerator(keras.utils.Sequence):

    def __init__(self, emb, counts, batchSize, augment=False):
        self.emb, self.counts = emb, counts
        self.batchSize = batchSize
        self.augment = augment

        self.rnd_size = int(np.ceil(self.batchSize * 0.2))
        self.rnd_vflip = np.random.randint(0, self.batchSize, self.rnd_size)
        self.rnd_hflip = np.random.randint(0, self.batchSize, self.rnd_size)
        self.rnd_bright = np.random.randint(0, self.batchSize, self.rnd_size)

        self.indices = np.arange(self.counts.shape[0])

    def __len__(self):
        return int(np.ceil(len(self.counts) / float(self.batchSize)))

    def __getitem__(self, idx):
        batch_emb = self.emb[idx * self.batchSize: (idx + 1) * self.batchSize, :, :, :]
        batch_counts = self.counts[idx * self.batchSize: (idx + 1) * self.batchSize]

        # randomly augment input data
        if self.augment:
            batch_emb[self.rnd_vflip] = np.flipud(batch_emb[self.rnd_vflip])
            batch_emb[self.rnd_hflip] = np.fliplr(batch_emb[self.rnd_hflip])
            batch_emb[self.rnd_bright] = batch_emb[self.rnd_bright] + np.random.uniform(-1, 1 )

        return (batch_emb, batch_counts)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


### build and compile model

# function that returns model
def build_and_compile_model():

    ## hyperparameters
    lr = 0.001 * hvd.size()
    tf.print('learning rate, adjusted by number of GPUS: ', lr)
    lamb = (1 / (2 * lr * epochs)) * 0.1
    tf.print('weight decay parameter: ', lamb)

    # starting filter value
    num_filters = 8
    kernel_size = 5
    # number and size of residual blocks
    num_blocks_list = [2, 5, 5, 2]
    dilation_rate_list = [1, 1, 1, 1]

    # build dense relu layers with batch norm and dropout
    def dense_layer(prev_layer, nodes, lamb=lamb):
        norm = layers.BatchNormalization(trainable=True)(prev_layer)
        dense = layers.Dense(nodes, activation='relu',
                             activity_regularizer=keras.regularizers.l2(lamb),
                             kernel_initializer=tf.keras.initializers.HeNormal())(norm)
        return dense

    # activation and batch normalization
    def bn_relu(inp_layer):
        bn = layers.BatchNormalization(trainable=True)(inp_layer)
        relu = layers.LeakyReLU()(bn)
        return relu

    # residual blocks (convolutions)
    def residual_block(inp_layer, downsample, filters, kernel_size, dilation_rate, lamb=lamb):
        y = layers.Conv2D(filters=filters,
                          kernel_size=kernel_size,
                          strides=(1 if not downsample else 2),
                          padding='same',
                          kernel_initializer=tf.keras.initializers.HeNormal(),
                          activity_regularizer=keras.regularizers.l2(lamb),
                          data_format='channels_first')(inp_layer)
        y = bn_relu(y)
        y = layers.Conv2D(filters=filters,
                          kernel_size=kernel_size,
                          strides=1,
                          dilation_rate=dilation_rate,
                          padding='same',
                          kernel_initializer=tf.keras.initializers.HeNormal(),
                          activity_regularizer=keras.regularizers.l2(lamb),
                          data_format='channels_first')(y)
        y = layers.BatchNormalization(trainable=True)(y)

        if downsample:
            inp_layer = layers.Conv2D(filters=filters,
                                      kernel_size=1,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=tf.keras.initializers.HeNormal(),
                                      activity_regularizer=keras.regularizers.l2(lamb),
                                      data_format='channels_first')(inp_layer)

        out = layers.Add()([inp_layer, y])
        out = layers.Activation('relu')(out)

        return out

    ## input
    tf.print('model input')
    inp = keras.Input(shape=(1, tokPerWindow, embeddingDim),
                      name='input')

    ## convolutional layers (ResNet)
    tf.print('residual blocks')

    # structure of residual blocks:
    # a) original: weight-BN-ReLU-weight-BN-addition-ReLU --> currently used
    # b) BN after addition: weight-BN-ReLU-weight-addition-BN-ReLU
    # c) ReLU before addition: weight-BN-ReLU-weight-BN-ReLU-addition
    # d) full pre-activation (SpliceAI): BN-ReLU-weight-BN-ReLU-weight-addition

    # initial convolution
    t = layers.BatchNormalization(trainable=True)(inp)
    t = layers.Conv2D(filters=num_filters,
                      kernel_size=kernel_size,
                      strides=2,
                      padding='same',
                      kernel_initializer=tf.keras.initializers.HeNormal(),
                      activity_regularizer=keras.regularizers.l2(lamb),
                      data_format='channels_first')(t)
    t = bn_relu(t)

    # residual blocks
    for i in range(len(num_blocks_list)):
        no_blocks = num_blocks_list[i]
        dil_rate = dilation_rate_list[i]

        t_shortcut = layers.Conv2D(filters=num_filters,
                                   kernel_size=kernel_size,
                                   strides=(1 if i == 0 else 2),
                                   padding='same',
                                   kernel_initializer=tf.keras.initializers.HeNormal(),
                                   activity_regularizer=keras.regularizers.l2(lamb),
                                   data_format='channels_first')(t)

        for j in range(no_blocks):
            t = residual_block(t,
                               downsample=(j == 0 and i != 0),
                               filters=num_filters,
                               kernel_size=kernel_size,
                               dilation_rate=dil_rate)

        t = layers.Add()([t, t_shortcut])
        num_filters *= 2

    t = layers.AveragePooling2D(pool_size=4,
                                data_format='channels_first',
                                padding='same')(t)
    flat = layers.Flatten()(t)

    ## dense layers
    tf.print('dense layers')
    # fully-connected layers with L2-regularization, batch normalization and dropout
    dense1 = dense_layer(flat, 512)

    out_norm = layers.BatchNormalization(trainable=True)(dense1)
    out = layers.Dense(1,
                       activation='linear',
                       kernel_initializer=tf.keras.initializers.HeNormal(),
                       activity_regularizer=keras.regularizers.l2(lamb),
                       name='output')(out_norm)

    ## concatenate to model
    model = keras.Model(inputs=inp, outputs=out)

    ## compile model
    tf.print('compile model')
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    opt = hvd.DistributedOptimizer(opt)

    model.compile(loss=keras.losses.MeanSquaredError(),
                  optimizer=opt,
                  metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'accuracy'],
                  experimental_run_tf_function=False)

    # for reproducibility during optimization
    tf.print('......................................................')
    tf.print('MIXTURE OF RESNET AND SPLICEAI')
    tf.print('optimizer: Adam')
    tf.print("learning rate: ", lr)
    tf.print('loss: mean squared error')
    tf.print('number/size of residual blocks: [2,5,5,2]')
    tf.print('dilation rates: [1,1,1,1]')
    tf.print('structure of residual block: a) original')
    tf.print('skip-connections between blocks')
    tf.print('channels: first')
    tf.print('activation function: leaky relu/he_normal')
    tf.print('starting filter value: ', num_filters)
    tf.print('kernel size: ', kernel_size)
    tf.print('number of dense layers before output layer: 1 (512, relu)')
    tf.print('output activation function: linear')
    tf.print('regularization: L2 - ', lamb)
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
                                      patience=epochs,
                                      min_delta=0.05,
                                      verbose=1,
                                      restore_best_weights=True)

callbacks = [
    es,
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),  # broadcast initial variables from rank 0 to all other servers
]

# save chackpoints only on worker 0
if hvd.rank() == 0:
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(filepath='/scratch2/hroetsc/Hotspots/results/model/best_model.h5',
                                           monitor='val_loss',
                                           mode='min',
                                           safe_best_only=True,
                                           verbose=1,
                                           save_weights_only=False))

# define number of steps - make sure that no. of steps is the same for all ranks!
# otherwise, stalled ranks problem might occur
steps = int(np.ceil(counts.shape[0] / batchSize))
val_steps = int(np.ceil(counts_test.shape[0] / batchSize))

# adjust by number of GPUs
steps = int(np.ceil(steps / hvd.size()))
val_steps = int(np.ceil(val_steps / hvd.size()))

## fit model
model = build_and_compile_model()

if hvd.rank() == 0:
    model.summary()
    print('train for {}, validate for {} steps per epoch'.format(steps, val_steps))
    print('using sequence generator')

fit = model.fit(x=SequenceGenerator(emb, counts, batchSize, augment=True),
                validation_data=SequenceGenerator(emb_test, counts_test, batchSize),
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
if hvd.rank() == 0:
    print('load best model')
    best_model = tf.keras.models.load_model('/scratch2/hroetsc/Hotspots/results/model/best_model.h5')

    ### MAIN PART ###
    # make prediction
    pred = best_model.predict(x=emb_test,
                              batch_size=batchSize,
                              verbose=1 if hvd.rank() == 0 else 0,
                              max_queue_size=256)
    print('counts:')
    print(counts_test)

    print('prediction:')
    print(pred.flatten())

    # merge actual and predicted counts
    prediction = pd.DataFrame({"Accession": tokens_test[:, 0],
                               "window": tokens_test[:, 1],
                               "label": labels_test,
                               "count": counts_test,
                               "pred_count": pred.flatten()})

    ### OUTPUT ###
    print('SAVE PREDICTED COUNTS')

    pd.DataFrame.to_csv(prediction,
                        '/scratch2/hroetsc/Hotspots/results/model_predictions_rank{}.csv'.format(int(hvd.rank())),
                        index=False)


tf.keras.backend.clear_session()
