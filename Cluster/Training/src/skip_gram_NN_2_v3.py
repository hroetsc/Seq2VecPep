### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  convert tokens to numerical vectors using a skip-gram neural network
# input:        word pairs (target and context word) generated in skip_gram_NN_1
# output:       embedded tokens (weights and their IDs)
# author:       HR

print("### PROTEOME EMBEDDING USING SKIP-GRAM NEURAL NETWORK - 2 ###")


import tensorflow as tf

import os
import sys
import json
import gc
import numpy as np
import pandas as pd

import keras

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.layers.core import Dense, Reshape
from keras.layers.embeddings import Embedding
from keras.layers import Dot, concatenate, merge, dot
from keras.layers import *
from keras.engine import input_layer

from keras.callbacks.callbacks import EarlyStopping
from keras.utils import Sequence

from keras import backend as K

from sklearn.model_selection import train_test_split

# overwrite namespace!
from tensorflow import keras
from tensorflow.keras import layers

# script from https://github.com/deepsense-ai/tensorflow_on_slurm
# distributed training using tensorflow with slurm
import tensorflow_on_slurm

# clean
gc.enable()
tf.keras.backend.clear_session()


### train on multiple GPUs and nodes

# log device placement (troubleshooting .....)
#tf.debugging.set_log_device_placement(True)
# must be at startup ...

#config = tf.compat.v1.ConfigProto()
#sess = tf.Session(config=config)

# select strategy and select implementation of collective ops
multiworker_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
    tf.distribute.experimental.CollectiveCommunication.AUTO)

#multiworker_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

#mirrored_strategy = tf.distribute.MirroredStrategy()

# from https://github.com/tensorflow/ecosystem/blob/master/distribution_strategy/keras_model_to_estimator.py
# dapted to use it with TF v.2.1.0
config = tf.estimator.RunConfig(tf_random_seed = 42,
                                train_distribute = multiworker_strategy,
                                eval_distribute = multiworker_strategy)

#sess = tf.Session(config=config)
#keras_estimator = tf.keras.estimator.model_to_estimator(
#  keras_model=model, config=config, model_dir=model_dir)


# nodes
nodelist = os.environ["SLURM_JOB_NODELIST"]
print("NODELIST: ")
print(nodelist)

nodename = os.environ["SLURMD_NODENAME"]
print("NODENAME: ")
print(nodename)

num_nodes = int(os.getenv("SLURM_JOB_NUM_NODES"))
print("NUMBER OF NODES: ")
print(num_nodes)


# specify cluster configuration
# strange, does not work at the moment

#cluster, my_job_name, my_task_index = tensorflow_on_slurm.tf_config_from_slurm(ps_number=1)
#cluster_spec = tf.train.ClusterSpec(cluster)
#server = tf.distribute.Server(server_or_cluster_def=cluster_spec,
#                                job_name=my_job_name,
#                                task_index=my_task_index)

# in case of parameter server
#if my_job_name == 'ps':
#    server.join()
#    sys.exit(0)


# allow memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus, 'GPU')

print("NUM GPUs AVAILABLE: ", len(tf.config.experimental.list_physical_devices('GPU')))


NUM_WORKERS = 20
GLOBAL_BATCH_SIZE = 32


# =============================================================================
# # HYPERPARAMETERS
# =============================================================================

#workers = 200

epochs = 500

valSplit = 0.1
#batchSize = 32

# =============================================================================
# # INPUT
# =============================================================================
print("LOAD DATA")

skip_grams = pd.read_csv(snakemake.input['skip_grams'], sep = " ", header = None)
ids = pd.read_csv(snakemake.input['ids'], header = None)
params = pd.read_csv('params.csv', header = 0)

output = snakemake.output['weights']

for i in range(len(output)):

    embeddingDim = int(params.iloc[i,0])

    print('-----------------------------------------------')
    print('EMBEDDING DIMENSIONS: ', embeddingDim)
    print('-----------------------------------------------')

    # =============================================================================
    # split skip-grams into target, context and label np.array()
    target_word = np.array(skip_grams.iloc[:,0], dtype = 'int32')
    context_word = np.array(skip_grams.iloc[:,1], dtype = 'int32')
    Y = np.array(skip_grams.iloc[:,2], dtype = 'int32')

    # =============================================================================
    # for testing!!!
    ind = np.random.randint(0, target_word.shape[0], 100000)
    target_word = target_word[ind]
    context_word = context_word[ind]
    Y = Y[ind]
    # =============================================================================

    print('target word vector')
    target_word = target_word.reshape(target_word.shape[0],)
    print(target_word)


    print('context word vector')
    context_word = context_word.reshape(context_word.shape[0],)
    print(context_word)


    print('label vector (converted 0 to -1)')
    Y = Y.reshape(Y.shape[0],)
    # replace 0 by -1
    Y = np.where(Y == 0, -1, Y)
    print(Y)


    # get vocabulary size
    vocab_size = len(ids.index) + 1
    print("vocabulary size (number of target word IDs + 1): {}".format(vocab_size))


    # =============================================================================
    # # MODEL CREATION
    # =============================================================================
    print("MODEL GENERATION")


    ### train on GPUs
    with multiworker_strategy.scope():

        input_target = keras.Input(((1,)), name='target_word')
        input_context = keras.Input(((1,)), name='context_word')

        # embed input layers
        embedding = layers.Embedding(input_dim = vocab_size,
                                output_dim = embeddingDim,
                                input_length = 1,
                                embeddings_initializer = 'he_uniform',
                                name = 'embedding')

        # apply embedding
        target = embedding(input_target)
        target = layers.Reshape((embeddingDim,1), name='target_embedding')(target) # every individual skip-gram has dimension embedding x 1
        context = embedding(input_context)
        context = layers.Reshape((embeddingDim,1), name='context_embedding')(context)

        # dot product similarity - normalize to get value between 0 and 1!
        dot_product = layers.dot([target, context], axes = 1, normalize = True, name = 'dot_product')
        dot_product = layers.Reshape((1,))(dot_product)

        # add dense layers
        x = layers.Dense(64, activation = 'tanh', kernel_initializer = 'he_uniform', name='1st_dense')(dot_product)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(1, activation = 'tanh', kernel_initializer = 'he_uniform', name='2nd_dense')(x)



        # create the primary training model
        model = keras.Model(inputs=[input_target, input_context], outputs=output)

        # do not specify adam decay and learning rate
        model.compile(loss=keras.losses.SquaredHinge(),
                        optimizer = keras.optimizers.Adam(),
                        metrics=[tf.keras.metrics.Accuracy()])


    # view model summary
    print(model.summary())


    # =============================================================================
    # # TRAINING
    # =============================================================================
    print("MODEL TRAINING")
    # split data into training and validation
    print("split word pairs into training and validation data sets")
    target_train, target_test, context_train, context_test, Y_train, Y_test = train_test_split(target_word, context_word, Y, test_size=valSplit)

    print('model metrics: {}'.format(model.metrics_names))


    # apply batch generator
    print("generating batches for model training")



    class BatchGenerator(keras.utils.Sequence):

         def __init__(self, target, context, Y, batch_size):
             self.target, self.context, self.Y = target, context, Y
             self.batch_size = batch_size
             self.indices = np.arange(self.target.shape[0])

         def __len__(self):
             return int(np.ceil(len(self.target) / float(self.batch_size)))

         def __getitem__(self, idx):

             batch_target = self.target[idx*self.batch_size : (idx + 1)*self.batch_size]
             batch_context = self.context[idx*self.batch_size : (idx + 1)*self.batch_size]
             batch_Y = self.Y[idx*self.batch_size : (idx + 1)*self.batch_size]

             return [batch_target, batch_context], batch_Y

         def on_epoch_end(self):
             np.random.shuffle(self.indices)


    train_generator = BatchGenerator(target_train, context_train, Y_train, GLOBAL_BATCH_SIZE)
    test_generator = BatchGenerator(target_test, context_test, Y_test, GLOBAL_BATCH_SIZE)

    # early stopping if model is already converged
    es = EarlyStopping(monitor = 'accuracy',
                        mode = 'max',
                        patience = 5,
                        min_delta = 0.003,
                        verbose = 1)

    # checkpoints
    ckpt = tf.keras.callbacks.ModelCheckpoint(filepath='/scratch2/hroetsc/Seq2Vec')

    # fit model
    print("fit the model")

    # fit_generator is also deprecated

    fit = model.fit_generator(generator = train_generator,
                        validation_data = test_generator,
                        steps_per_epoch = target_train.shape[0],
                        validation_steps = target_test.shape[0],
                        validation_freq = 1,
                        epochs = epochs,
                        callbacks = [es, ckpt],
                        initial_epoch = 0,
                        verbose = 2,
                        max_queue_size = 1,
                        workers = NUM_WORKERS,
                        use_multiprocessing = True,
                        shuffle = True)

    # =============================================================================
    # ### OUTPUT ###
    # =============================================================================

    print("SAVE WEIGHTS")
    # get word embedding

    print("configuration of embedding layer:")
    print(model.layers[2].get_config())
    weights = model.layers[2].get_weights()[0] # weights of the embedding layer of target word

    # save weights of embedding matrix
    df = pd.DataFrame(weights)
    pd.DataFrame.to_csv(df, snakemake.output['weights'][i], header=False)
    df.head()

    # save model
    model.save(snakemake.output['model'][i])

    # save accuracy and loss
    m = open(snakemake.output['metrics'][i], 'w')
    m.write("accuracy \t {} \n val_accuracy \t {} \n loss \t {} \n val_loss \t {}".format(fit.history['accuracy'], fit.history['val_accuracy'], fit.history['loss'], fit.history['val_loss']))
    m.close()

    # clear session
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
