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

from D_helper import print_initialization,\
    format_input, open_and_format_matrices,\
    RestoreBestModel, lr_schedule, LearningRateScheduler,\
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

    theta = 0.5
    tf.print('compression factor: ', theta)
    k = 6
    tf.print('growth rate: ', k)
    blocks = 3
    tf.print('number of dense blocks: ', blocks)
    n_layers_list = [4, 4, 4]
    n_filters_list = [8, 8, 8]

    # composite function
    def conv(layer, maps, strides, bottleneck=True):
        if bottleneck:
            bn = layers.BatchNormalization(trainable=True)(layer)
            act = layers.Activation('relu')(bn)
            layer = layers.Conv2D(filters=maps,
                                  kernel_size=(1, 1),
                                  strides=strides,
                                  padding='same',
                                  data_format='channels_first')(act)

        bn = layers.BatchNormalization(trainable=True)(layer)
        act = layers.Activation('relu')(bn)
        layer = layers.Conv2D(filters=maps,
                              kernel_size=(3, 3),
                              strides=strides,
                              padding='same',
                              data_format='channels_first')(act)
        return layer

    # dense block
    def dense_block(layer, n_layers, maps, k=k):
        for i in range(n_layers):
            conv_out = conv(layer, maps=maps, strides=1)
            layer = layers.Concatenate(axis=1)([layer, conv_out])
            maps += k

        return layer, maps

    # transition layer with compression
    def transition(layer, theta=theta):
        m = layer.shape[1]
        n_maps = int(np.floor(m * theta))

        bn = layers.BatchNormalization(trainable=True)(layer)
        act = layers.Activation('relu')(bn)
        layer = layers.Conv2D(filters=n_maps,
                              kernel_size=(1, 1),
                              strides=1,
                              padding='same',
                              data_format='channels_first')(act)

        pool = layers.AveragePooling2D(pool_size=(2, 2),
                                       strides=2,
                                       padding='same',
                                       data_format='channels_first')(layer)
        return pool

    ## input
    tf.print('model input')
    inp0 = keras.Input(shape=(1, tokPerWindow, embeddingDim),
                       name='input')

    # initial convolution
    inp = layers.BatchNormalization(trainable=True)(inp0)
    inp = layers.Activation('relu')(inp)
    inp = layers.Conv2D(filters=8,
                        kernel_size=(7, 7),
                        strides=1,
                        padding='same',
                        data_format='channels_first')(inp)
    layer = layers.MaxPool2D(pool_size=(3, 3),
                             strides=1,
                             padding='same',
                             data_format='channels_first')(inp)
    tf.print(layer.shape)

    for b in range(blocks):
        layer, maps = dense_block(layer, n_layers=n_layers_list[b], maps=n_filters_list[b])
        layer = transition(layer)
        tf.print(layer.shape)

    layer = layers.GlobalAveragePooling2D(data_format='channels_first')(layer)
    tf.print(layer.shape)

    bn = layers.BatchNormalization(trainable=True)(layer)
    fc = layers.Dense(1024,
                      activation='relu',
                      kernel_initializer=tf.keras.initializers.HeNormal())(bn)
    bn = layers.BatchNormalization(trainable=True)(fc)
    out = layers.Dense(1,
                       activation='linear')(bn)

    ## concatenate to model
    model = keras.Model(inputs=inp0, outputs=out)

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
    tf.print('DENSE-NET')
    tf.print('optimizer: Adam')
    tf.print("learning rate: ", lr)
    tf.print('loss: mean squared error')
    tf.print('activity regularization: none')
    tf.print('using batch normalization: yes')
    tf.print('using Dropout layer: no')
    tf.print('......................................................')

    return model


#### train model
print('MODEL TRAINING')

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
                max_queue_size=1,
                verbose=2 if hvd.rank() == 0 else 0,
                shuffle=True)

### OUTPUT ###
print('SAVE MODEL AND METRICS')
save_training_res(model, fit)


########## part 2: make prediction ##########
print('MAKE PREDICTION')
### INPUT ###
if hvd.rank() == 0:
    ### MAIN PART ###
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

    ### OUTPUT ###
    print('SAVE PREDICTED COUNTS')

    pd.DataFrame.to_csv(prediction,
                        '/scratch2/hroetsc/Hotspots/results/model_predictions.csv',
                        index=False)

tf.keras.backend.clear_session()
