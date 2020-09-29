### HEADER ###
# HOTSPOT PREDICTION
# description: set up neural network to predict hotspot density along the human proteome
# input: table with tokens in sliding windows and scores,
# output: model, metrics, predictions for test data set
# author: HR


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
    lamb = (1 / (2 * lr * epochs)) * 0.0001
    tf.print('weight decay parameter: ', lamb)

    # build dense relu layers with batch norm and dropout
    def dense_layer(prev_layer, nodes, lamb=lamb):
        norm = layers.BatchNormalization(trainable=True)(prev_layer)
        dense = layers.Dense(nodes, activation='selu',
                             activity_regularizer=tf.keras.regularizers.l1(lamb),
                             kernel_initializer=tf.keras.initializers.LecunNormal())(norm)
        return dense

    # residual blocks (convolutions)
    def residual_block(prev_layer, no_filters, kernel_size, strides, lamb=lamb):
        conv = layers.Conv2D(filters=no_filters,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding='same',
                             kernel_initializer=tf.keras.initializers.HeNormal(),
                             activity_regularizer=tf.keras.regularizers.l1(lamb),
                             data_format='channels_first')(prev_layer)

        norm = layers.BatchNormalization(trainable=True)(conv)
        act = layers.LeakyReLU()(norm)

        pool = layers.AveragePooling2D(pool_size=2,
                                       strides=2,
                                       data_format='channels_first',
                                       padding='same')(act)
        return pool

    ## input
    tf.print('model input')
    inp = keras.Input(shape=(1, tokPerWindow, embeddingDim),
                      name='input')

    ## hyperparameters
    num_filters = 16  # starting filter value
    kernel_size = 5
    strides = 1

    # ## convolutional layers
    tf.print('convolutional layers')
    conv1 = residual_block(inp, num_filters, kernel_size, strides)
    conv2 = residual_block(conv1, int(num_filters * 2), kernel_size, strides)
    conv3 = residual_block(conv2, int(num_filters * 4), kernel_size, strides)

    ## dense layers
    tf.print('dense layers')
    flat = layers.Flatten()(conv3)
    dense1 = dense_layer(flat, 1024)
    dense2 = dense_layer(dense1, 256)
    dense3 = dense_layer(dense2, 64)

    out_norm = layers.BatchNormalization(trainable=True)(dense3)
    out = layers.Dense(1, activation='linear',
                       kernel_initializer=tf.keras.initializers.HeNormal(),
                       activity_regularizer=tf.keras.regularizers.l1(lamb),
                       name='output')(out_norm)

    ## concatenate to model
    model = keras.Model(inputs=inp, outputs=out)

    ## compile model
    tf.print('compile model')
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    opt = hvd.DistributedOptimizer(opt)

    model.compile(loss='mean_squared_error',
                  optimizer=opt,
                  metrics=['mean_absolute_percentage_error', 'mean_absolute_error'],
                  experimental_run_tf_function=False)

    # for reproducibility during optimization
    tf.print('......................................................')
    tf.print('SIMPLE CONVOLUTIONAL STRUCTURE')
    tf.print('optimizer: Adam')
    tf.print("learning rate: ", lr)
    tf.print('loss: mean squared error')
    tf.print('channels: first')
    tf.print('pooling: Average2D, strides = 2')
    tf.print('activation function: leaky relu / he_normal')
    tf.print('number of dense layers before output layer: 3 (1024-64, selu)')
    tf.print('output activation function: linear')
    tf.print('starting filter value: ', num_filters)
    tf.print('kernel size: ', kernel_size)
    tf.print('strides: ', strides)
    tf.print('number of convolutions: 3')
    tf.print('activity regularization: L1 - ', lamb)
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
                max_queue_size=256,
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
