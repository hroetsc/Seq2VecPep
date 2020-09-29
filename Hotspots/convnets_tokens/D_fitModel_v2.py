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
                             activity_regularizer=keras.regularizers.l1(lamb),
                             kernel_initializer=tf.keras.initializers.HeNormal())(norm)
        return dense

    # activation and batch normalization
    def bn_relu(inp_layer):
        bn = layers.BatchNormalization(trainable=True)(inp_layer)
        relu = layers.Activation('relu')(bn)
        return relu

    # residual blocks (convolutions)
    def residual_block(inp_layer, downsample, filters, kernel_size, dilation_rate, lamb=lamb):
        y = layers.Conv2D(filters=filters,
                          kernel_size=kernel_size,
                          strides=(1 if not downsample else 2),
                          padding='same',
                          kernel_initializer=tf.keras.initializers.HeNormal(),
                          activity_regularizer=keras.regularizers.l1(lamb),
                          data_format='channels_first')(inp_layer)
        y = bn_relu(y)
        y = layers.Conv2D(filters=filters,
                          kernel_size=kernel_size,
                          strides=1,
                          dilation_rate=dilation_rate,
                          padding='same',
                          kernel_initializer=tf.keras.initializers.HeNormal(),
                          activity_regularizer=keras.regularizers.l1(lamb),
                          data_format='channels_first')(y)
        y = layers.BatchNormalization(trainable=True)(y)

        if downsample:
            inp_layer = layers.Conv2D(filters=filters,
                                      kernel_size=1,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=tf.keras.initializers.HeNormal(),
                                      activity_regularizer=keras.regularizers.l1(lamb),
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
                      strides=1,
                      padding='same',
                      activity_regularizer=keras.regularizers.l1(lamb),
                      kernel_initializer=tf.keras.initializers.HeNormal(),
                      data_format='channels_first')(t)
    t = bn_relu(t)

    # residual blocks
    for i in range(len(num_blocks_list)):
        no_blocks = num_blocks_list[i]
        dil_rate = dilation_rate_list[i]
        for j in range(no_blocks):
            t = residual_block(t,
                               downsample=(j == 0 and i != 0),
                               filters=num_filters,
                               kernel_size=kernel_size,
                               dilation_rate=dil_rate)
        num_filters *= 2

    t = layers.AveragePooling2D(pool_size=4,
                                data_format='channels_first',
                                padding='same')(t)
    flat = layers.Flatten()(t)

    ## dense layers
    tf.print('dense layers')
    # fully-connected layers with L2-regularization, batch normalization and dropout
    dense1 = dense_layer(flat, 256)

    out_norm = layers.BatchNormalization(trainable=True)(dense1)
    out = layers.Dense(1, activation='linear',
                       activity_regularizer=keras.regularizers.l1(lamb),
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
                  metrics=['accuracy', 'mean_absolute_error', 'mean_absolute_percentage_error'],
                  experimental_run_tf_function=False)

    # for reproducibility during optimization
    tf.print('......................................................')
    tf.print('RESNET STRUCTURE')
    tf.print('optimizer: Adam')
    tf.print("learning rate: ", lr)
    tf.print('number/size of residual blocks: [2,5,5,2]')
    tf.print('dilation rates: [1,1,1,1]')
    tf.print('structure of residual block: a) original')
    tf.print('channels: first')
    tf.print('activation function: relu/he_normal')
    tf.print('number of dense layers before output layer: 1 (256, relu)')
    tf.print('output activation function: linear')
    tf.print('end filter value: ', num_filters)
    tf.print('kernel size: ', kernel_size)
    tf.print('regularization: L1 - ', lamb)
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
save_training_res(model)

########## part 2: make prediction ##########
print('MAKE PREDICTION')
### INPUT ###

# create ensemble of nets from all ranks and get common prediction
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
