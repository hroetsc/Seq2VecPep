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

from D_helper import print_initialization, \
    format_input, open_and_format_matrices, \
    RestoreBestModel, CosineAnnealing, \
    save_training_res, combine_predictions

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

epochs = 20
pseudocounts = 1
no_features = int(tokPerWindow * 10)
augment = False

no_cycles = int(epochs / 5)

print('number of epochs: ', epochs)
print('number of learning rate cycles: ', no_cycles)
print('number of pseudocounts: ', pseudocounts)
print('using scaled input data: False')
print('selecting ', no_features, ' best features (not for prediction)')
print('training data augmentation: ', augment)
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

proteins = pd.read_csv('/scratch2/hroetsc/Hotspots/data/sequence_embedings.csv')

### MAIN PART ###
print('FORMAT INPUT AND GET EMBEDDING MATRICES')

tokens, labels, counts = format_input(tokensAndCounts_train)
tokens, labels, counts, emb, bestFeatures, pca = open_and_format_matrices(tokens, labels, counts, emb_train, acc_train,
                                                                          no_features=no_features,
                                                                          proteins=proteins,
                                                                          augment=augment, ret_pca=True)

tokens_test, labels_test, counts_test = format_input(tokensAndCounts_test)
tokens_test, labels_test, counts_test, emb_test, \
bestFeatures_test, pca_test = open_and_format_matrices(tokens_test,
                                                       labels_test,
                                                       counts_test,
                                                       emb_test,
                                                       acc_test,
                                                       no_features=no_features,
                                                       proteins=proteins,
                                                       augment=False,
                                                       ret_pca=True)


### build and compile model
# function that returns model
def build_and_compile_model(max_lr, starting_filter, kernel_size, num_blocks, block_size, dense_nodes):
    ## hyperparameters
    max_lr = max_lr * hvd.size()

    # number and size of residual blocks
    num_blocks_list = [block_size] * num_blocks
    dilation_rate_list = [1] * num_blocks

    # build dense relu layers with batch norm and dropout
    def dense_layer(prev_layer, dense_nodes):
        norm = layers.BatchNormalization(trainable=True)(prev_layer)
        dense = layers.Dense(dense_nodes, activation='relu',
                             kernel_initializer=tf.keras.initializers.HeNormal())(norm)
        return dense

    # activation and batch normalization
    def bn_relu(inp_layer):
        bn = layers.BatchNormalization(trainable=True)(inp_layer)
        relu = layers.LeakyReLU()(bn)
        return relu

    # residual blocks (convolutions)
    def residual_block(inp_layer, downsample, filters, kernel_size, dilation_rate):
        y = layers.Conv2D(filters=filters,
                          kernel_size=kernel_size,
                          strides=(1 if not downsample else 2),
                          padding='same',
                          kernel_initializer=tf.keras.initializers.HeNormal(),
                          data_format='channels_first')(inp_layer)
        y = bn_relu(y)
        y = layers.Conv2D(filters=filters,
                          kernel_size=kernel_size,
                          strides=1,
                          dilation_rate=dilation_rate,
                          padding='same',
                          kernel_initializer=tf.keras.initializers.HeNormal(),
                          data_format='channels_first')(y)
        y = layers.BatchNormalization(trainable=True)(y)

        if downsample:
            inp_layer = layers.Conv2D(filters=filters,
                                      kernel_size=1,
                                      strides=2,
                                      padding='same',
                                      kernel_initializer=tf.keras.initializers.HeNormal(),
                                      data_format='channels_first')(inp_layer)

        out = layers.Add()([inp_layer, y])
        out = layers.Activation('relu')(out)

        return out

    ## input
    tf.print('model input')
    inp = keras.Input(shape=(2, tokPerWindow, embeddingDim),
                      name='input')

    ## convolutional layers (ResNet)
    tf.print('residual blocks')

    # initial convolution
    t = layers.BatchNormalization(trainable=True)(inp)
    t = layers.Conv2D(filters=starting_filter,
                      kernel_size=kernel_size,
                      strides=2,
                      padding='same',
                      kernel_initializer=tf.keras.initializers.HeNormal(),
                      data_format='channels_first')(t)
    t = bn_relu(t)

    # residual blocks
    for i in range(len(num_blocks_list)):
        no_blocks = num_blocks_list[i]
        dil_rate = dilation_rate_list[i]

        t_shortcut = layers.Conv2D(filters=starting_filter,
                                   kernel_size=kernel_size,
                                   strides=(1 if i == 0 else 2),
                                   padding='same',
                                   kernel_initializer=tf.keras.initializers.HeNormal(),
                                   data_format='channels_first')(t)

        for j in range(no_blocks):
            t = residual_block(t,
                               downsample=(j == 0 and i != 0),
                               filters=starting_filter,
                               kernel_size=kernel_size,
                               dilation_rate=dil_rate)

        t = layers.Add()([t, t_shortcut])
        starting_filter *= 2

    t = layers.AveragePooling2D(pool_size=4,
                                data_format='channels_first',
                                padding='same')(t)
    flat = layers.Flatten()(t)

    ## dense layers
    tf.print('dense layers')
    # fully-connected layer
    dense = dense_layer(flat, dense_nodes)

    out_norm = layers.BatchNormalization(trainable=True)(dense)
    out = layers.Dense(1, activation='linear',
                       kernel_initializer=tf.keras.initializers.GlorotUniform(),
                       name='output')(out_norm)

    ## concatenate to model
    model = keras.Model(inputs=inp, outputs=out)

    ## compile model
    tf.print('compile model')
    opt = tf.keras.optimizers.Adam(learning_rate=max_lr)
    opt = hvd.DistributedOptimizer(opt)

    model.compile(loss=keras.losses.MeanSquaredError(),
                  optimizer=opt,
                  metrics=['mean_absolute_error', 'mean_absolute_percentage_error', 'accuracy'],
                  experimental_run_tf_function=False)

    # for reproducibility during optimization
    tf.print('......................................................')
    tf.print('MIXTURE OF RESNET AND SPLICEAI')
    tf.print('optimizer: Adam')
    tf.print('loss: mean squared error')
    tf.print('skip-connections between blocks')
    tf.print('channels: first')
    tf.print('activation function: leaky relu/he_normal')
    tf.print('regularization: none')
    tf.print('using batch normalization: yes')
    tf.print('using Dropout layer: no')
    tf.print('......................................................')

    return model


#### train model
print('MODEL TRAINING')


def fitness(batch_size, max_lr, starting_filter, kernel_size, num_blocks, block_size, dense_nodes):

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    model = build_and_compile_model(max_lr=max_lr * hvd.size(), starting_filter=starting_filter,
                                    kernel_size=kernel_size,
                                    num_blocks=num_blocks, block_size=block_size, dense_nodes=dense_nodes)

    # define callbacks
    callbacks = [RestoreBestModel(),
                 CosineAnnealing(no_cycles=no_cycles, no_epochs=epochs, max_lr=max_lr * hvd.size()),
                 hvd.callbacks.BroadcastGlobalVariablesCallback(0)]

    batchSize = batch_size * hvd.size()

    # define number of steps - make sure that no. of steps is the same for all ranks!
    # otherwise, stalled ranks problem might occur
    steps = int(np.ceil(counts.shape[0] / batchSize))
    val_steps = int(np.ceil(counts_test.shape[0] / batchSize))

    # adjust by number of GPUs
    steps = int(np.ceil(steps / hvd.size()))
    val_steps = int(np.ceil(val_steps / hvd.size()))

    if hvd.rank() == 0:
        model.summary()
        print('train for {}, validate for {} steps per epoch'.format(steps, val_steps))

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
                    max_queue_size=64,
                    verbose=2 if hvd.rank() == 0 else 0,
                    shuffle=True)

    loss = min(fit.history['val_loss'])
    loss_str = str(loss).replace(".", "")

    # predictions
    pred = model.predict(x=emb_test,
                         batch_size=batchSize,
                         verbose=1 if hvd.rank() == 0 else 0,
                         max_queue_size=1)

    # model metrics
    val = []
    name = list(fit.history.keys())
    for i, elem in enumerate(fit.history.keys()):
        val.append(fit.history[elem])

    m = list(zip(name, val))
    m = pd.DataFrame(m)

    if hvd.rank() == 0:
        print('#######################################################################################################')
        print('#######################################################################################################')
        print('LOCALS:')
        print(locals())
        print('#######################################################################################################')
        print('VALIDATION LOSS: ', loss)
        print('#######################################################################################################')
        print('#######################################################################################################')

        pd.DataFrame.to_csv(m, '/scratch2/hroetsc/Hotspots/results/opt_v4_metrics_{}.txt'.format(loss_str),
                            header=False,
                            index=False)

        # merge actual and predicted counts
        prediction = pd.DataFrame({"Accession": tokens_test[:, 0],
                                   "window": tokens_test[:, 1],
                                   "label": labels_test,
                                   "count": counts_test,
                                   "pred_count": pred.flatten()})

        pd.DataFrame.to_csv(prediction,
                            '/scratch2/hroetsc/Hotspots/results/opt_v4_prediction_{}.csv'.format(loss_str),
                            index=False)

        del model

        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()

    return loss


max_lrs = [1e-06, 5e-06, 1e-05, 5e-05, 1e-04, 5e-04, 1e-03, 5e-03, 1e-02, 5e-02]
filter_sizes = [8, 16, 32]
kernel_sizes = [3, 5, 7]
block_sizes = [4, 5, 6]
dense_nodes = [1024, 512]

for a in dense_nodes:
    for b in block_sizes:
        for c in kernel_sizes:
            for d in filter_sizes:
                for e in max_lrs:
                    fitness(batch_size=16, max_lr=e, starting_filter=d, kernel_size=c,
                            num_blocks=4, block_size=b, dense_nodes=a)

tf.keras.backend.clear_session()
