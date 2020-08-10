### HEADER ###
# HOTSPOT PREDICTION
# description: set up neural network to predict hotspot density along the human proteome
# input: table with tokens in sliding windows and scores, table with tf-idf weighted embeddings
# output: model
# author: HR

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

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
print('rank of the current process: ', hvd.rank()) # node
print('local rank of the current process: ', hvd.local_rank()) # process on node
print('Is MPI multi-threading supported? ', hvd.mpi_threads_supported())
print('Is MPI enabled? ', hvd.mpi_enabled())
print('Is Horovod compiled with MPI? ', hvd.mpi_built())
print('Is Horovod compiled with NCCL? ', hvd.nccl_built())

print("-------------------------------------------------------------------------")


### initialize GPU training environment

# pin GPUs (each GPU gets single process)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


########## part 1: fit model ##########
### INPUT ###
tokensAndCounts_train = pd.read_csv('/scratch2/hroetsc/Hotspots/data/windowTokens_training.csv')
weights = pd.read_csv('/scratch2/hroetsc/Hotspots/data/token_embeddings.csv')
tfidf_train = pd.read_csv('/scratch2/hroetsc/Hotspots/data/TFIDF_training.csv')

# tokensAndCounts_train = pd.read_csv('results/windowTokens_training.csv')
# weights = pd.read_csv('results/token_embeddings.csv')
# tfidf_train = pd.read_csv('results/TFIDF_training.csv')

### HYPERPARAMETERS ###
embeddingDim = 100
tokPerWindow = 8

dt_size = int(1e02)
#dt_size = len(tokensAndCounts_train)

#epochs = 20
epochs = 100
batchSize = 16

epochs = int(np.ceil(epochs/hvd.size()))
batchSize = batchSize*hvd.size()

print('number of epochs, adjusted by number of GPUs: ', epochs)
print('batch size, adjusted by number of GPUs: ', batchSize)


### MAIN PART ###
### format input

def format_input(tokensAndCounts, sz):

    tokens = np.array(tokensAndCounts.loc[:int(sz), ['Accession', 'tokens']], dtype='object')
    counts = np.array(tokensAndCounts.loc[:int(sz), ['counts']], dtype='float64')

    return tokens, counts

tokens, counts = format_input(tokensAndCounts_train, dt_size)


#### batch generator
# generates sequence representation matrices on the fly


def findEmbeddings(batch_token, tfidf):

    tokens_split = [str.split(' ') for str in [batch_token[1]]][0]
    acc = batch_token[0]

    out = [None] * tokPerWindow

    for n, t in enumerate(tokens_split):
        # find embedding
        emb = np.array(weights[weights['subword'] == t].iloc[:, 1:(embeddingDim + 1)]).flatten()
        # find tf-idf
        tf_idf = float(tfidf[(tfidf['Accession'] == acc) & (tfidf['token'] == t)]['tf_idf'])

        # multiply embeddings by tf-idf score
        out[n] = tf_idf * emb

    return np.array(out)



class SequenceGenerator(keras.utils.Sequence):

    def __init__(self, tokens, counts, batchSize):

        self.tokens, self.counts = tokens, counts
        self.batchSize = batchSize
        self.indices = np.arange(self.counts.shape[0])

    def __len__(self):
        return int(np.ceil(len(self.counts) / float(self.batchSize)))

    def __getitem__(self, idx):
        batch_counts = self.counts[idx * self.batchSize: (idx + 1) * self.batchSize]
        batch_tokens = self.tokens[idx * self.batchSize: (idx + 1) * self.batchSize, ]

        batch_embeddings = [findEmbeddings(batch_token, tfidf_train) for batch_token in list(batch_tokens)]

        return (np.array(batch_embeddings), batch_counts)

    def on_epoch_end(self):
        pass
        #np.random.shuffle(self.indices)


### build and compile model
def build_and_compile_model():

    ## layers
    tf.print('model input')
    inp = keras.Input(shape=(tokPerWindow, embeddingDim, 1),
                      name='input')

    # first convolution
    tf.print('convolutions')
    conv1 = layers.Conv2D(32, 2, 2,
                          activation='relu',
                          padding='same',
                          kernel_regularizer=keras.regularizers.l2(l=0.01),
                          data_format='channels_first',
                          name='conv1')(inp)
    pool1 = layers.MaxPool2D((2, 2),
                             name='pool1')(conv1)

    # second convolution
    conv2 = layers.Conv2D(64, 2, 2,
                          activation='relu',
                          padding='same',
                          kernel_regularizer=keras.regularizers.l2(l=0.01),
                          data_format='channels_first',
                          name='conv2')(pool1)
    pool2 = layers.MaxPool2D((2, 2),
                             name='pool2')(conv2)

    # third convolution
    conv3 = layers.Conv2D(128, 2, 2,
                          activation='relu',
                          padding='same',
                          kernel_regularizer=keras.regularizers.l2(l=0.01),
                          data_format='channels_first',
                          name='conv3')(pool2)
    pool3 = layers.MaxPool2D((2, 2),
                             name='pool3')(conv3)

    # flatten
    tf.print('flatten and pass to fully connected layers')
    flat = layers.Flatten()(pool3)

    # fully connected layer with L2-regularization
    fully = layers.Dense(128, activation='relu',
                         kernel_regularizer=keras.regularizers.l2(l=0.01),
                         name='relu')(flat)

    # dropout and output layer
    drop = layers.Dropout(0.2)(fully)
    out = layers.Dense(1, activation='linear',
                       name='output')(drop) # linear activation bc no binary classification

    ## model
    model = keras.Model(inputs=inp, outputs=out)

    ## compile model
    tf.print('compile model')
    opt = tf.keras.optimizers.Adam(learning_rate=0.01 * hvd.size())
    opt = hvd.DistributedOptimizer(opt)

    model.compile(loss=keras.losses.MeanSquaredError(),
                  optimizer=opt,
                  metrics=['mean_absolute_error', 'mean_absolute_percentage_error', 'cosine_proximity'],
                  experimental_run_tf_function=False)

    return model


#### train model
# define callbacks - adapt later for multi-node training
callbacks = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0), # broadcast initial variables from rank 0 to all other servers
]

# save chackpoints only on worker 0
if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath = '/scratch2/hroetsc/Hotspots/results/model/ckpts',
                                                monitor = 'val_loss',
                                                mode = 'min',
                                                safe_best_only = False,
                                                verbose = 1,
                                                save_weights_only=True))


# split data into training and validation
tr = int(np.ceil(counts.shape[0]*0.9))
tokens_train = tokens[:tr, ]
tokens_val = tokens[(tr+1):, ]

counts_train = counts[:tr]
counts_val = counts[(tr+1):]


# define number of steps - make sure that no. of steps is the same for all ranks!
# otherwise, stalled ranks problem might occur
steps = int(np.ceil(counts_train.shape[0] / batchSize))
val_steps = int(np.ceil(counts_val.shape[0] / batchSize))

# adjust by number of GPUs
steps = int(np.ceil(steps / hvd.size()))
val_steps = int(np.ceil(val_steps / hvd.size()))


# fit model
model = build_and_compile_model()
model.summary()

print('train for {} steps, validate for {} steps per epoch'.format(steps, val_steps))

# do not use generator
# emb_train = np.array([findEmbeddings(batch_token, tfidf_train) for batch_token in list(tokens_train)])
# emb_val = np.array([findEmbeddings(batch_token, tfidf_train) for batch_token in list(tokens_val)])

# SequenceGenerator(tokens_train, counts_train, batchSize)
# SequenceGenerator(tokens_val, counts_val, batchSize)

model.fit(SequenceGenerator(tokens_train, counts_train, batchSize),
          validation_data=SequenceGenerator(tokens_val, counts_val, batchSize),
          batch_size=batchSize,
          steps_per_epoch=steps,
          validation_steps=val_steps,
          epochs=epochs,
          callbacks=callbacks,
          initial_epoch=0,
          max_queue_size=256,
          verbose=2,
          shuffle=False)


### OUTPUT ###
if hvd.rank() == 0:
    # save weights
    model.save_weights('/scratch2/hroetsc/Hotspots/results/weights.h5')
    # save entire model
    model.save('/scratch2/hroetsc/Hotspots/results/model.h5')

    # save metrics
    val = []
    name = list(fit.history.keys())
    for i, elem in enumerate(fit.history.keys()):
        val.append(fit.history[elem])

    m = list(zip(name, val))
    m = pd.DataFrame(m)
    pd.DataFrame.to_csv(m, '/scratch2/hroetsc/Hotspots/results/model_metrics.txt', header=False, index=False)


########## part 2: make prediction ##########
### INPUT ###
tokensAndCounts_test = pd.read_csv('/scratch2/hroetsc/Hotspots/data/windowTokens_testing.csv')
tfidf_test = pd.read_csv('/scratch2/hroetsc/Hotspots/data/TFIDF_testing.csv')

tokens_test, counts_test = format_input(tokensAndCounts_test, len(tokensAndCounts_test))

### MAIN PART ###
pred = model.predict(x=np.array([findEmbeddings(tokens_test, tfidf_test)]),
                     batch_size=batchSize)

prediction = pd.DataFrame({"count": counts_test,
                           "prediction": pred})

### OUTPUT ###
if hvd.rank() == 0:
    pd.DataFrame.to_csv(prediction, '/scratch2/hroetsc/Hotspots/results/model_predictions.csv', index=False)


tf.keras.backend.clear_session()
