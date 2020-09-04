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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.keras.backend.clear_session()
import horovod.tensorflow.keras as hvd

hvd.init()

from D_helper import print_initialization, \
    format_input, open_and_format_matrices, \
    RestoreBestModel, lr_schedule, LearningRateScheduler, \
    save_training_res

print_initialization()

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

epochs = 160
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
mu = pd.read_csv('/scratch2/hroetsc/Hotspots/data/mean_emb.csv')
mu = np.tile(np.array(mu).flatten(), tokPerWindow).reshape((tokPerWindow, embeddingDim))

tokensAndCounts_train = pd.read_csv('/scratch2/hroetsc/Hotspots/data/windowTokens_OPTtraining.csv')
emb_train = '/scratch2/hroetsc/Hotspots/data/embMatrices_OPTtraining.dat'
acc_train = '/scratch2/hroetsc/Hotspots/data/embMatricesAcc_OPTtraining.dat'

tokensAndCounts_test = pd.read_csv('/scratch2/hroetsc/Hotspots/data/windowTokens_OPTtesting.csv')
emb_test = '/scratch2/hroetsc/Hotspots/data/embMatrices_OPTtesting.dat'
acc_test = '/scratch2/hroetsc/Hotspots/data/embMatricesAcc_OPTtesting.dat'

### MAIN PART ###
print('FORMAT INPUT AND GET EMBEDDING MATRICES')

tokens, labels, counts = format_input(tokensAndCounts_train)
tokens, labels, counts, emb = open_and_format_matrices(tokens, labels, counts, emb_train, acc_train, mu, augment=True)

tokens_test, labels_test, counts_test = format_input(tokensAndCounts_test)
tokens_test, labels_test, counts_test, emb_test = open_and_format_matrices(tokens_test, labels_test, counts_test,
                                                                           emb_test, acc_test, mu)


### build and compile model

# function that returns model
def build_and_compile_model():
    ## hyperparameters
    lr = 0.001 * hvd.size()
    tf.print('learning rate, adjusted by number of GPUS: ', lr)

    def bn_relu(layer):
        bn = layers.BatchNormalization(trainable=True)(layer)
        relu = layers.Activation('relu')(bn)
        return relu

    def convolution(layer, filters, kernel_size, strides, pool_size=2):
        conv = layers.Conv2D(filters=filters,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding='same',
                             activation='relu',
                             data_format='channels_first')(layer)
        pool = layers.MaxPool2D(pool_size=pool_size,
                                strides=strides,
                                padding='same',
                                data_format='channels_first')(conv)
        return pool

    def convolutional_block(layer, filters, kernel_size):
        ### BLOCK A ###
        A_init = convolution(inp, filters=filters, kernel_size=1, strides=1, pool_size=1)
        # a) original: weight-BN-ReLU-weight-BN-addition-ReLU
        A = layers.Conv2D(filters=filters,
                          kernel_size=(kernel_size, kernel_size),
                          strides=1,
                          padding='same',
                          data_format='channels_first')(A_init)
        A = bn_relu(A)
        A = layers.Conv2D(filters=filters,
                          kernel_size=(kernel_size, kernel_size),
                          strides=1,
                          padding='same',
                          data_format='channels_first')(A)
        A = layers.BatchNormalization(trainable=True)(A)
        A_add = layers.Add()([A_init, A])
        A_add = layers.Activation('relu')(A_add)
        A_add = convolution(A_add, filters=filters, kernel_size=kernel_size, strides=2)


        ### BLOCK B ###
        B_init = convolution(inp, filters=filters, kernel_size=1, strides=1, pool_size=1)
        # b) BN after addition: weight-BN-ReLU-weight-addition-BN-ReLU
        B = layers.Conv2D(filters=filters,
                          kernel_size=(kernel_size, kernel_size),
                          strides=1,
                          padding='same',
                          data_format='channels_first')(B_init)
        B = bn_relu(B)
        B = layers.Conv2D(filters=filters,
                          kernel_size=(kernel_size, kernel_size),
                          strides=1,
                          padding='same',
                          data_format='channels_first')(B)
        B_add = layers.Add()([B_init, B])
        B_add = bn_relu(B_add)
        B_add = convolution(B_add, filters=filters, kernel_size=kernel_size, strides=2)


        ### BLOCK C ###
        C_init = convolution(inp, filters=filters, kernel_size=1, strides=1, pool_size=1)
        # c) ReLU before addition: weight-BN-ReLU-weight-BN-ReLU-addition
        C = layers.Conv2D(filters=filters,
                          kernel_size=(kernel_size, kernel_size),
                          strides=1,
                          padding='same',
                          data_format='channels_first')(C_init)
        C = bn_relu(C)
        C = layers.Conv2D(filters=filters,
                          kernel_size=(kernel_size, kernel_size),
                          strides=1,
                          padding='same',
                          data_format='channels_first')(C)
        C = bn_relu(C)
        C_add = layers.Add()([C_init, C])
        C_add = convolution(C_add, filters=filters, kernel_size=kernel_size, strides=2)


        ### BLOCK D ###
        D_init = convolution(inp, filters=filters, kernel_size=1, strides=1, pool_size=1)
        # d) full pre-activation (SpliceAI): BN-ReLU-weight-BN-ReLU-weight-addition
        D = bn_relu(D_init)
        D = layers.Conv2D(filters=filters,
                          kernel_size=(kernel_size, kernel_size),
                          strides=1,
                          padding='same',
                          data_format='channels_first')(D)
        D = bn_relu(D)
        D = layers.Conv2D(filters=filters,
                          kernel_size=(kernel_size, kernel_size),
                          strides=1,
                          padding='same',
                          data_format='channels_first')(D)
        D_add = layers.Add()([D_init, D])
        D_add = convolution(D_add, filters=filters, kernel_size=kernel_size, strides=2)

        ### concatenate ###
        conc = layers.Concatenate(axis=1)([A_add, B_add, C_add, D_add])
        flat = layers.Flatten()(conc)

        return flat


    ## input
    tf.print('model input')
    inp = keras.Input(shape=(1, tokPerWindow, embeddingDim),
                      name='input')

    block = convolutional_block(inp, filters=16, kernel_size=3)

    dense = layers.Dense(1024, activation='relu')(block)
    dense = layers.Dense(256, activation='relu')(dense)
    dense = layers.Dense(64, activation='relu')(dense)

    out_norm = layers.BatchNormalization(trainable=True)(dense)
    out = layers.Dense(1,
                       activation='linear',
                       kernel_initializer=tf.keras.initializers.HeNormal(),
                       name='output')(out_norm)

    ## concatenate to model
    model = keras.Model(inputs=inp, outputs=out)

    ## compile model
    tf.print('compile model')
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    opt = hvd.DistributedOptimizer(opt)

    model.compile(loss=keras.losses.MeanSquaredError(),
                  optimizer=opt,
                  metrics=['mean_absolute_error', 'mean_absolute_percentage_error', 'accuracy'],
                  experimental_run_tf_function=False)

    # for reproducibility during optimization
    tf.print('......................................................')
    tf.print('MIXTURE OF RESNET AND SPLICEAI')
    tf.print('optimizer: Adam')
    tf.print("learning rate: ", lr)
    tf.print('loss: mean squared error')
    tf.print('regularization: none')
    tf.print('using batch normalization: yes')
    tf.print('using Dropout layer: no')
    tf.print('......................................................')

    return model


#### train model
print('MODEL TRAINING')
# define callbacks
callbacks = [RestoreBestModel(),
             LearningRateScheduler(schedule=lr_schedule, compress=.4),
             hvd.callbacks.BroadcastGlobalVariablesCallback(0),
             tf.keras.callbacks.ModelCheckpoint(
                 filepath='/scratch2/hroetsc/Hotspots/results/model/model_rank{}.h5'.format(hvd.rank()),
                 monitor='val_loss',
                 mode='min',
                 safe_best_only=False,
                 verbose=1,
                 save_weights_only=False)]

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

fit = model.fit(x=emb,
                y=counts,
                batch_size=batchSize,
                validation_data=(emb_test, counts_test),
                validation_batch_size=batchSize,
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
save_training_res(model, fit)

########## part 2: make prediction ##########
print('MAKE PREDICTION')

if hvd.rank() == 0:
    # make prediction
    pred = model.predict(x=emb_test,
                         batch_size=16,
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

    print('SAVE PREDICTED COUNTS')
    pd.DataFrame.to_csv(prediction,
                        '/scratch2/hroetsc/Hotspots/results/model_predictions.csv',
                        index=False)

tf.keras.backend.clear_session()
