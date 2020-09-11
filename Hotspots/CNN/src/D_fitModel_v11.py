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

epochs = 400
batchSize = 32
pseudocounts = 1
no_cycles = 2
no_features = int(tokPerWindow * 20)
augment = False

epochs = int(np.ceil(epochs / hvd.size()))
batchSize = batchSize * hvd.size()

print('number of epochs, adjusted by number of GPUs: ', epochs)
print('batch size, adjusted by number of GPUs: ', batchSize)
print('number of learning rate cycles: ', no_cycles)
print('number of pseudocounts: ', pseudocounts)
print('using scaled input data: True --> using biophysical properties as features')
print('selecting ', no_features, ' best features for prediction')
print('using augmented training data: ', augment)
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
print('no training data augmentation')

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
def build_and_compile_model():
    ## hyperparameters
    lr = 0.001 * hvd.size()
    tf.print('learning rate, adjusted by number of GPUS: ', lr)

    drop_prob = .3
    tf.print('dropout probability: ', drop_prob)

    kernel_size = 1
    tf.print('kernel size: ', kernel_size)

    def bn_relu(layer, nodes):
        norm = layers.BatchNormalization()(layer)
        drop = layers.Dropout(drop_prob)(norm)
        dense = layers.Dense(nodes, activation='relu',
                             kernel_initializer=keras.initializers.GlorotUniform())(drop)
        return dense

    ## model
    inp0 = layers.Input(shape=(1, tokPerWindow, embeddingDim),
                        name='input_window')
    inp1 = layers.Input(shape=(1, embeddingDim),
                        name='input_protein')

    conv = layers.Conv2D(filters=32,
                         kernel_size=kernel_size,
                         strides=2,
                         padding='same',
                         kernel_initializer=keras.initializers.HeNormal(),
                         bias_initializer=keras.initializers.Zeros(),
                         data_format='channels_first')(inp0)
    norm = layers.BatchNormalization()(conv)
    act = layers.Activation('relu')(norm)
    pool = layers.MaxPool2D(padding='same',
                            data_format='channels_first')(act)
    pool = layers.Flatten()(pool)
    dense = bn_relu(pool, 128)

    conc = layers.Concatenate()([inp1, dense])
    dense = bn_relu(conc, 2048)
    dense = bn_relu(dense, 1024)
    dense = bn_relu(dense, 128)

    out = layers.Dense(1, activation='linear',
                       kernel_initializer=keras.initializers.GlorotUniform(),
                       bias_initializer=keras.initializers.Zeros(),
                       use_bias=True,
                       name='output')(dense)

    ## concatenate to model
    model = keras.Model(inputs=[inp0, inp1], outputs=out)

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
    tf.print('RESTART FROM SCRATCH (AGAIN)')
    tf.print('optimizer: Adam')
    tf.print("learning rate: --> cosine annealing", lr)
    tf.print('loss: mean squared error')
    tf.print('regularization: none')
    tf.print('using batch normalization: yes')
    tf.print('using Dropout layer: yes')
    tf.print('he normal initialization for relu layers')
    tf.print('output layer: bias, linear activation, glorot uniform initialization')
    tf.print('......................................................')

    return model


#### train model
print('MODEL TRAINING')
# define callbacks
callbacks = [RestoreBestModel(),
             CosineAnnealing(no_cycles=no_cycles, no_epochs=epochs, max_lr=0.01),
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

fit = model.fit(x=[emb, prots],
                y=counts,
                batch_size=batchSize,
                validation_data=(emb_test, counts_test),
                validation_batch_size=batchSize,
                steps_per_epoch=steps,
                validation_steps=val_steps,
                epochs=epochs,
                callbacks=callbacks,
                initial_epoch=1,
                max_queue_size=256,
                verbose=2 if hvd.rank() == 0 else 0,
                shuffle=True)

### OUTPUT ###
print('SAVE MODEL AND METRICS')
save_training_res(model, fit)

########## part 2: make prediction ##########
print('MAKE PREDICTION')
# make prediction
pred = model.predict(x=[emb_test, prots_test],
                     batch_size=4,
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
                    '/scratch2/hroetsc/Hotspots/results/model_prediction_rank{}.csv'.format(hvd.rank()),
                    index=False)

if hvd.rank() == 0:
    combine_predictions()

tf.keras.backend.clear_session()
