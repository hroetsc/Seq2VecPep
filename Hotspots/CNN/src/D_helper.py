import os
import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
import horovod.tensorflow.keras as hvd

########################################################################################################################
# INITIALIZATION
########################################################################################################################

def print_initialization():

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
    print('rank of the current process: ', hvd.rank())  # node
    print('local rank of the current process: ', hvd.local_rank())  # process on node
    print('Is MPI multi-threading supported? ', hvd.mpi_threads_supported())
    print('Is MPI enabled? ', hvd.mpi_enabled())
    print('Is Horovod compiled with MPI? ', hvd.mpi_built())
    print('Is Horovod compiled with NCCL? ', hvd.nccl_built())

    print("-------------------------------------------------------------------------")


########################################################################################################################
# HYPERPARAMETERS
########################################################################################################################

embeddingDim = 128
tokPerWindow = 8

epochs = 160
batchSize = 16
pseudocounts = 1

epochs = int(np.ceil(epochs / hvd.size()))
batchSize = batchSize * hvd.size()


########################################################################################################################
# DATA FORMATTING
########################################################################################################################

def format_input(tokensAndCounts):
    tokens = np.array(tokensAndCounts.loc[:, ['Accession', 'tokens']], dtype='object')
    counts = np.array(tokensAndCounts['counts'], dtype='float32')

    # log-transform counts (+ pseudocounts)
    counts = np.log2((counts + pseudocounts))

    # get binary labels
    labels = np.where(counts == 0, 0, 1)

    print('number of features: ', counts.shape[0])

    return tokens, labels, counts



def open_and_format_matrices(tokens, labels, counts, emb_path, acc_path, mu, augment=False):
    no_elements = int(tokPerWindow * embeddingDim)  # number of matrix elements per sliding window
    # how many bytes are this? (32-bit encoding --> 4 bytes per element)
    chunk_size = int(no_elements * 4)

    embMatrix = [None] * tokens.shape[0]
    accMatrix = [None] * tokens.shape[0]
    chunk_pos = 0

    # open weights and accessions binary file
    with open(emb_path, 'rb') as emin, open(acc_path, 'rb') as ain:
        # loop over files to get elements
        for b in range(tokens.shape[0]):
            emin.seek(chunk_pos, 0)  # set cursor position with respect to beginning of file
            # read current chunk of embeddings and format in matrix shape
            dt = np.fromfile(emin, dtype='float32', count=no_elements)

            # normalize input data (z-transformation)
            # dt = (dt - np.mean(dt)) / np.std(dt)

            # make sure to pass 4D-Tensor to model: (batchSize, depth, height, width)
            dt = dt.reshape((tokPerWindow, embeddingDim))

            # for 2D convolution --> 5d input:
            embMatrix[b] = np.expand_dims(dt, axis=0)
            # embMatrix[b] = dt

            # get current accession (index)
            ain.seek(int(b * 4), 0)
            accMatrix[b] = int(np.fromfile(ain, dtype='int32', count=1))

            # increment chunk position
            chunk_pos += chunk_size

        emin.close()
        ain.close()

    # order tokens and count according to order in embedding matrix
    accMatrix = np.array(accMatrix, dtype='int32')
    tokens = tokens[accMatrix, :]
    counts = counts[accMatrix]
    labels = labels[accMatrix]

    embMatrix = np.array(embMatrix, dtype='float32')

    # randomly augment training data
    if augment:
        rnd_size = int(np.ceil(embMatrix.shape[0] * 0.15))
        rnd_vflip = np.random.randint(0, embMatrix.shape[0], rnd_size)
        rnd_hflip = np.random.randint(0, embMatrix.shape[0], rnd_size)
        rnd_bright = np.random.randint(0, embMatrix.shape[0], rnd_size)

        embMatrix_vflip = np.flipud(embMatrix[rnd_vflip, :, :, :])
        embMatrix_hflip = np.fliplr(embMatrix[rnd_hflip, :, :, :])
        embMatrix_bright = embMatrix[rnd_bright, :, :, :] + np.random.uniform(-1, 1)

        embMatrix = np.concatenate((embMatrix, embMatrix_vflip, embMatrix_hflip, embMatrix_bright))
        labels = np.concatenate((labels, labels[rnd_vflip], labels[rnd_hflip], labels[rnd_bright]))
        counts = np.concatenate((counts, counts[rnd_vflip], counts[rnd_hflip], counts[rnd_bright]))
        tokens = np.concatenate((tokens, tokens[rnd_vflip], tokens[rnd_hflip], tokens[rnd_bright]))

    # output: reformatted tokens and counts, embedding matrix
    return tokens, labels, counts, embMatrix


########################################################################################################################
# CALLBACKS
########################################################################################################################

class RestoreBestModel(keras.callbacks.Callback):
    def __init__(self):
        super(RestoreBestModel, self).__init__()
        self.best_weights = None  # best weights

    def on_train_begin(self, logs=None):
        self.best = np.Inf  # initialize best as Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('val_loss')  # get validation loss
        if np.less(current, self.best):
            self.best = current
            self.best_weights = self.model.get_weights()  # record the best weights

    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)
        print('RESTORING WEIGHTS FROM VALIDATION LOSS {}'.format(self.best))


# learning rate schedule
# triangular learning rate function
def triang(x, amp, period):
    return (2*amp / np.pi)*np.arcsin(np.sin(((2*np.pi)/period)*x))+amp

# plan for learning rates
e = np.arange(0,epochs+1)
LR_SCHEDULE = [(x, triang(x, 0.005, int(epochs/5))) for x in e]

def lr_schedule(epoch, cnt_lr):
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return cnt_lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return cnt_lr

class LearningRateScheduler(keras.callbacks.Callback):
    def __init__(self, schedule, compress):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.compress = compress

    def on_train_begin(self, logs=None):
        self.val_loss = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        cnt_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        if np.greater(logs.get('val_loss'), self.val_loss):
            scheduled_lr = self.schedule(epoch, cnt_lr)*self.compress
        else:
            scheduled_lr = self.schedule(epoch, cnt_lr)

        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        self.val_loss = logs.get('val_loss')
        print("epoch {}: learning rate is {}".format(epoch, scheduled_lr))


########################################################################################################################
# SAVE METRICS AND PREDICTION
########################################################################################################################

def save_training_res(model, fit):
    # save entire model
    model.save('/scratch2/hroetsc/Hotspots/results/model/best_model_rank{}.h5'.format(hvd.rank()))

    if hvd.rank() == 0:
        # save weights
        model.save_weights('/scratch2/hroetsc/Hotspots/results/model/weights.h5')

        # save metrics
        val = []
        name = list(fit.history.keys())
        for i, elem in enumerate(fit.history.keys()):
            val.append(fit.history[elem])

        m = list(zip(name, val))
        m = pd.DataFrame(m)
        pd.DataFrame.to_csv(m, '/scratch2/hroetsc/Hotspots/results/model_metrics.txt', header=False, index=False)
