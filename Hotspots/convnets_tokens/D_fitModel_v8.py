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
    drop_prob = 0.3
    tf.print('dropout probability: ', drop_prob)
    lr = 0.001 * hvd.size()
    tf.print('learning rate, adjusted by number of GPUS: ', lr)
    lamb = (1 / (2 * lr * epochs)) * 0.0001
    tf.print('weight decay parameter: ', lamb)

    ## functions
    def dense_layer(prev_layer, nodes, drop_prob=drop_prob, dropout=False, lamb=lamb):

        if dropout:
            norm = layers.Dropout(drop_prob)(prev_layer)
            norm = layers.BatchNormalization(trainable=True)(norm)
        else:
            norm = layers.BatchNormalization(trainable=True)(prev_layer)

        dense = layers.Dense(nodes, activation='relu',
                             kernel_initializer=tf.keras.initializers.HeNormal(),
                             activity_regularizer=tf.keras.regularizers.l1(lamb))(norm)
        return dense

    def convolution(prev_layer, no_filters, kernel_size, strides, pooling=True, lamb=lamb):
        conv = layers.LocallyConnected2D(filters=no_filters,
                                         kernel_size=kernel_size,
                                         strides=strides,
                                         padding='valid',
                                         kernel_initializer=tf.keras.initializers.HeNormal(),
                                         activity_regularizer=tf.keras.regularizers.l1(lamb),
                                         data_format='channels_first')(prev_layer)

        norm = layers.BatchNormalization(trainable=True)(conv)
        act = layers.Activation('relu')(norm)

        if pooling:
            pool = layers.MaxPool2D(pool_size=2,
                                    strides=2,
                                    data_format='channels_first',
                                    padding='valid')(act)
            return pool

        else:
            return act

    ## input
    tf.print('model input')

    inp = keras.Input(shape=(1, tokPerWindow, embeddingDim),
                      name='input')

    dense = dense_layer(inp, embeddingDim)
    dense = layers.Flatten()(dense)

    no_dense = 6
    tf.print('number of dense layers: ', no_dense)

    nodes = 2048
    for l in range(no_dense):
        dense = dense_layer(dense, nodes)
        nodes *= 0.5

    res = layers.Reshape((1, 8, 8))(dense)
    conv = convolution(res, no_filters=8, kernel_size=3, strides=1)
    conv = layers.Flatten()(conv)

    conv = dense_layer(conv, 1024)
    conv = dense_layer(conv, 256)
    conv = dense_layer(conv, 64)

    # classification task
    class_norm = layers.BatchNormalization(trainable=True)(dense)
    classification = layers.Dense(1, activation='sigmoid',
                                  kernel_initializer=tf.keras.initializers.HeNormal(),
                                  activity_regularizer=tf.keras.regularizers.l1(lamb),
                                  name='classification')(class_norm)

    # regression task
    reg_norm = layers.BatchNormalization(trainable=True)(conv)
    regression = layers.Dense(1, activation='linear',
                              kernel_initializer=tf.keras.initializers.HeNormal(),
                              activity_regularizer=tf.keras.regularizers.l1(lamb),
                              name='regression')(reg_norm)

    ## concatenate to model
    model = keras.Model(inputs=inp, outputs=[classification, regression])

    ## compile model
    tf.print('compile model')

    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    opt = hvd.DistributedOptimizer(opt)

    losses = {'classification': 'binary_crossentropy',
              'regression': 'mean_squared_error'}

    loss_weights = {'classification': 0.5,
                    'regression': 1.0}
    tf.print(loss_weights.keys())
    tf.print(loss_weights.values())

    metrics = {'classification': ['binary_accuracy'],
               'regression': ['mean_absolute_error', 'mean_absolute_percentage_error', 'cosine_proximity']}

    model.compile(loss=losses,
                  loss_weights=loss_weights,
                  optimizer=opt,
                  metrics=metrics,
                  experimental_run_tf_function=False)

    # for reproducibility during optimization
    tf.print('......................................................')
    tf.print('BINARY CLASSIFICATION AND REGRESSION AT THE SAME TIME')
    tf.print('optimizer: Adam - ', lr)
    tf.print('loss: binary_crossentropy and mean_squared_error')
    tf.print('activation function dense layers: relu/he normal')
    tf.print('activity regularization: L1 - ', lamb)
    tf.print('kernel initialization: he normal')
    tf.print('using batch normalization: yes')
    tf.print('using Dropout layer: yes, ', drop_prob)
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

fit = model.fit(x=emb,
                y=[labels, counts],
                batch_size=batchSize,
                validation_data=(emb_test, [labels_test, counts_test]),
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
    print(pred[1].flatten())

    # merge actual and predicted counts
    prediction = pd.DataFrame({"Accession": tokens_test[:, 0],
                               "window": tokens_test[:, 1],
                               "label": labels_test,
                               "count": counts_test,
                               "pred_label": pred[0].flatten(),
                               "pred_count": pred[1].flatten()})

    ### OUTPUT ###
    print('SAVE PREDICTED COUNTS')

    pd.DataFrame.to_csv(prediction,
                        '/scratch2/hroetsc/Hotspots/results/model_predictions.csv',
                        index=False)

tf.keras.backend.clear_session()
