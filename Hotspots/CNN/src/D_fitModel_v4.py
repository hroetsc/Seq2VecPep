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
    save_training_res, combine_predictions, ensemble_predictions

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

epochs = 100
batchSize = 16
pseudocounts = 1
no_features = int(tokPerWindow * 10)
augment = False

# epochs = int(np.ceil(epochs / hvd.size()))
batchSize = batchSize * hvd.size()
no_cycles = int(epochs / 10)

max_lr = 0.001
starting_filter = 8
kernel_size = 3
block_size = 5
num_blocks = 4
dense_nodes = 1024

print('number of epochs, adjusted by number of GPUs: ', epochs)
print('batch size, adjusted by number of GPUs: ', batchSize)
print('number of learning rate cycles: ', no_cycles)
print('maximal learning rate in schedule', max_lr)
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
tokens, labels, counts, emb, prots, bestFeatures, pca = open_and_format_matrices(tokens, labels, counts, emb_train,
                                                                                 acc_train,
                                                                                 no_features=no_features,
                                                                                 proteins=proteins,
                                                                                 augment=augment, ret_pca=True)

tokens_test, labels_test, counts_test = format_input(tokensAndCounts_test)
tokens_test, labels_test, counts_test, emb_test, prots_test, \
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
def build_and_compile_model(max_lr, starting_filter, kernel_size, block_size, dense_nodes, num_blocks):
    num_blocks_list = [block_size] * num_blocks
    dilation_rate_list = [1] * num_blocks

    print(locals())

    # build dense relu layers with batch norm and dropout
    def dense_layer(prev_layer, nodes):
        norm = layers.BatchNormalization(trainable=True)(prev_layer)
        dense = layers.Dense(nodes, activation='relu',
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
    inp_window = keras.Input(shape=(1, tokPerWindow, embeddingDim),
                             name='input_window')

    inp_prots = keras.Input(shape=(1, tokPerWindow, embeddingDim),
                            name='input_prots')

    inp = layers.Concatenate(axis=1)([inp_window, inp_prots])

    ## convolutional layers (ResNet)
    tf.print('residual blocks')

    # structure of residual blocks:
    # a) original: weight-BN-ReLU-weight-BN-addition-ReLU --> currently used
    # b) BN after addition: weight-BN-ReLU-weight-addition-BN-ReLU
    # c) ReLU before addition: weight-BN-ReLU-weight-BN-ReLU-addition
    # d) full pre-activation (SpliceAI): BN-ReLU-weight-BN-ReLU-weight-addition

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
    dense1 = dense_layer(flat, dense_nodes)

    out_norm = layers.BatchNormalization(trainable=True)(dense1)
    out = layers.Dense(1,
                       activation='linear',
                       kernel_initializer=tf.keras.initializers.GlorotUniform(),
                       name='output')(out_norm)

    ## concatenate to model
    model = keras.Model(inputs=[inp_window, inp_prots], outputs=out)

    ## compile model
    tf.print('compile model')
    opt = tf.keras.optimizers.Adam(learning_rate=max_lr * hvd.size())
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
# define callbacks
callbacks = [RestoreBestModel(),
             CosineAnnealing(no_cycles=no_cycles, no_epochs=epochs, max_lr=max_lr * hvd.size()),
             hvd.callbacks.BroadcastGlobalVariablesCallback(0)]

# define number of steps - make sure that no. of steps is the same for all ranks!
# otherwise, stalled ranks problem might occur
steps = int(np.ceil(counts.shape[0] / batchSize))
val_steps = int(np.ceil(counts_test.shape[0] / batchSize))

# adjust by number of GPUs
steps = int(np.ceil(steps / hvd.size()))
val_steps = int(np.ceil(val_steps / hvd.size()))

## fit model
model = build_and_compile_model(max_lr=max_lr, starting_filter=starting_filter, kernel_size=kernel_size,
                                block_size=block_size, dense_nodes=dense_nodes, num_blocks=num_blocks)

if hvd.rank() == 0:
    model.summary()
    print('train for {}, validate for {} steps per epoch'.format(steps, val_steps))

fit = model.fit(x=[emb, prots],
                y=counts,
                batch_size=batchSize,
                validation_data=([emb_test, prots_test], counts_test),
                validation_batch_size=batchSize,
                steps_per_epoch=steps,
                validation_steps=val_steps,
                epochs=epochs,
                callbacks=callbacks,
                initial_epoch=0,
                max_queue_size=64,
                verbose=2 if hvd.rank() == 0 else 0,
                shuffle=True)

### OUTPUT ###
print('SAVE MODEL AND METRICS')
save_training_res(model, fit)

########## part 2: make prediction ##########
print('MAKE PREDICTION')
# make prediction
pred = model.predict(x=[emb_test, prots_test],
                     batch_size=batchSize,
                     verbose=1 if hvd.rank() == 0 else 0,
                     max_queue_size=64)
print('counts:')
print(counts_test)

print('prediction:')
pred_counts = np.array(pred.flatten())
print(pred_counts)

# merge actual and predicted counts
prediction = pd.DataFrame({"Accession": tokens_test[:, 0],
                           "window": tokens_test[:, 1],
                           "label": labels_test,
                           "count": counts_test,
                           "pred_count": pred_counts})

print('SAVE PREDICTED COUNTS')
pd.DataFrame.to_csv(prediction,
                    '/scratch2/hroetsc/Hotspots/results/model_prediction_rank{}.csv'.format(hvd.rank()),
                    index=False)

if hvd.rank() == 0:
    combine_predictions()

    ensemble_prediction = ensemble_predictions(emb_test, prots_test, batchSize)
    prediction = pd.DataFrame({"Accession": tokens_test[:, 0],
                               "window": tokens_test[:, 1],
                               "label": labels_test,
                               "count": counts_test,
                               "pred_count": ensemble_prediction})
    pd.DataFrame.to_csv(prediction,
                        '/scratch2/hroetsc/Hotspots/results/ensemble_prediction.csv',
                        index=False)

tf.keras.backend.clear_session()
