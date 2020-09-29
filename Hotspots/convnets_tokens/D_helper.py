import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import *
import tensorflow as tf
from tensorflow import keras
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


########################################################################################################################
# DATA FORMATTING
########################################################################################################################

def format_input(tokensAndCounts):
    print('format tokens and counts and retrieve labels')

    tokens = np.array(tokensAndCounts.loc[:, ['Accession', 'tokens']], dtype='object')
    counts = np.array(tokensAndCounts['counts'], dtype='float32')

    # log-transform counts (+ pseudocounts)
    counts = np.log2((counts + pseudocounts))

    # get binary labels
    labels = np.where(counts == 0, 0, 1)

    print('number of features: ', counts.shape[0])

    return tokens, labels, counts


def open_and_format_matrices(tokens, labels, counts, emb_path, acc_path, no_features, proteins,
                             augment=False, ret_pca=False):
    print('open and format embedding matrices, get all features')

    no_elements = int(tokPerWindow * embeddingDim)  # number of matrix elements per sliding window
    # how many bytes are this? (32-bit encoding --> 4 bytes per element)
    chunk_size = int(no_elements * 4)

    embMatrix = [None] * tokens.shape[0]
    PCAs = [None] * tokens.shape[0]
    accMatrix = [None] * tokens.shape[0]
    chunk_pos = 0

    pca = PCA(n_components=tokPerWindow)

    def scaling(x, a, b):
        sc_x = (x - np.min(x)) / (np.max(x) - np.min(x))
        return (b - a) * sc_x + a

    # open weights and accessions binary file
    with open(emb_path, 'rb') as emin, open(acc_path, 'rb') as ain:
        # loop over files to get elements
        for b in range(tokens.shape[0]):
            emin.seek(chunk_pos, 0)  # set cursor position with respect to beginning of file
            # read current chunk of embeddings and format in matrix shape
            dt = np.fromfile(emin, dtype='float32', count=no_elements)

            # get current accession (index)
            ain.seek(int(b * 4), 0)
            cnt_acc = int(np.fromfile(ain, dtype='int32', count=1))
            accMatrix[b] = cnt_acc

            # for biophysical properties: replace NaNs
            # dt = np.nan_to_num(dt)
            # z-transform input data
            # dt = (dt - np.min(dt)) / (np.max(dt) - np.min(dt))

            # scale input data between 0 and 255 (color)
            # dt = scaling(dt, a=0, b=255)

            # make sure to pass 4D-Tensor to model: (batchSize, depth, height, width)
            dt = dt.reshape((tokPerWindow, embeddingDim))

            # for 2D convolution --> 5d input:
            embMatrix[b] = np.expand_dims(dt, axis=0)
            # embMatrix[b] = dt

            # apply PCA
            PCAs[b] = np.expand_dims(pca.fit_transform(dt), axis=0)

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
    PCAs = np.array(PCAs)

    # get protein embeddings
    tok = pd.DataFrame(tokens[:, 0], columns=['Accession'])
    prots = tok.merge(proteins, how='left')
    prots = np.array(prots.iloc[:, 1:])

    # get the proper shape and merge with embedding matrix
    # prots = np.array([np.tile(prots[i, :], tokPerWindow).reshape((tokPerWindow, embeddingDim)) for i in range(prots.shape[0])])
    # prots = np.expand_dims(prots, axis=1)

    # embMatrix = np.concatenate((embMatrix, prots), axis=1)

    # # multiply with embedding matrix
    # embMatrix = np.array([embMatrix[i, :, :, :]*prots[i, :] for i in range(embMatrix.shape[0])])

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
        PCAs = np.concatenate((PCAs, PCAs[rnd_vflip], PCAs[rnd_hflip],
                               PCAs[rnd_bright]))  # should actually be augmented too! -- keep in mind if applying
        labels = np.concatenate((labels, labels[rnd_vflip], labels[rnd_hflip], labels[rnd_bright]))
        counts = np.concatenate((counts, counts[rnd_vflip], counts[rnd_hflip], counts[rnd_bright]))
        tokens = np.concatenate((tokens, tokens[rnd_vflip], tokens[rnd_hflip], tokens[rnd_bright]))

    # select best features
    embMatrix_flat = np.array([embMatrix[i, :, :, :].flatten() for i in range(embMatrix.shape[0])])
    selector = SelectKBest(f_regression, k=no_features)
    BestFeatures = selector.fit_transform(X=embMatrix_flat, y=counts)

    # output: reformatted tokens and counts, embedding matrix
    if ret_pca:
        return tokens, labels, counts, embMatrix, prots, BestFeatures, PCAs

    else:
        return tokens, labels, counts, embMatrix, prots, BestFeatures


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


# plan for learning rates
class CosineAnnealing(keras.callbacks.Callback):
    def __init__(self, no_cycles, no_epochs, max_lr):
        super(CosineAnnealing, self).__init__()
        self.no_cycles = no_cycles
        self.no_epochs = no_epochs
        self.max_lr = max_lr
        self.lrates = list()
        self.ensemble = 0

    def cos_annealing(self, epoch, no_cycles, no_epochs, max_lr):
        epochs_per_cycle = np.floor(no_epochs / no_cycles)
        cos_arg = (np.pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
        return (max_lr / 2) * (np.cos(cos_arg) + 1)

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.cos_annealing(epoch, self.no_cycles, self.no_epochs, self.max_lr)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.lrates.append(lr)

    def on_epoch_end(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

        if (epoch+1) % self.no_cycles == 0:
            self.model.save(
                '/scratch2/hroetsc/Hotspots/results/model/ensemble_rank{}_epoch{}.h5'.format(hvd.rank(), self.ensemble))
            self.ensemble += 1
            print('saving model for ensemble prediction')

        print("epoch {}: learning rate is {}".format(epoch, lr))


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


def combine_predictions():
    print('combining predictions from all ranks')

    for i in range(hvd.size()):
        cnt_pred = pd.read_csv('/scratch2/hroetsc/Hotspots/results/model_prediction_rank{}.csv'.format(i))
        if i == 0:
            res = pd.DataFrame({'Accession': cnt_pred['Accession'],
                                'window': cnt_pred['window'],
                                'label': cnt_pred['label'],
                                'count': cnt_pred['count'],
                                'pred_count': cnt_pred['pred_count'] * (1 / hvd.size())})
        else:
            res['pred_count'] += cnt_pred['pred_count'] * (1 / hvd.size())

    pd.DataFrame.to_csv(res,
                        '/scratch2/hroetsc/Hotspots/results/model_predictions.csv',
                        index=False)


def ensemble_predictions(emb_test, prots_test, batchSize):
    print('using learning rate schedule minima for ensemble prediction')

    ranks = hvd.size()
    models_per_rank = int(epochs / no_cycles)

    all_predictions = []

    for i in range(ranks):
        for j in range(models_per_rank):

            cnt_model = keras.models.load_model(
                '/scratch2/hroetsc/Hotspots/results/model/ensemble_rank{}_epoch{}.h5'.format(i, j))
            cnt_prediction = cnt_model.predict(x=[emb_test, prots_test],
                                               batch_size=batchSize,
                                               verbose=1 if hvd.rank() == 0 else 0,
                                               max_queue_size=64)

            all_predictions.append(cnt_prediction.flatten())

    # average of all predictions
    all_predictions = np.array(all_predictions)
    average_prediction = np.mean(all_predictions, axis=0)

    return average_prediction