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

from skopt.space import Real, Categorical, Integer

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

epochs = 10
pseudocounts = 1
no_features = int(tokPerWindow * 10)
augment = False

no_cycles = int(epochs / 5)
no_opt_iterations = 120

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

### hyperparameters
batch_size = Categorical([2, 4, 8], name='batch_size')
max_lr = Real(low=1e-07, high=0.05, name='max_lr')
n_conv_kernels = Categorical([32, 64, 128, 256], name='n_conv_kernels')
conv_kernel_size = Categorical([1, 3, 5], name='conv_kernel_size')
n_primary_caps = Integer(low=4, high=64, name='n_primary_caps')
dim_primary_caps = Integer(low=8, high=32, name='dim_primary_caps')
r = Integer(low=3, high=6, name='r')
units_1 = Categorical([64, 128, 256, 512], name='units_1')
units_2 = Categorical([256, 512, 1024, 2048], name='units_2')
drop_prob = Real(low=0.1, high=0.5, name='drop_prob')

dimensions = [batch_size,
              max_lr,
              n_conv_kernels,
              conv_kernel_size,
              n_primary_caps,
              dim_primary_caps,
              r,
              units_1,
              units_2,
              drop_prob]

default_parameters = [8, 0.00001, 32, 5, 32, 16, 5, 128, 512, 0.2]


### build and compile model
# write model
## LAYERS
# squash function: non-linear activation
def squash(s, epsilon=0, axis=-1):
    s_norm = tf.norm(s, axis=axis, keepdims=True)
    return (tf.square(s_norm) / (1 + tf.square(s_norm))) * (s / (s_norm + epsilon))  # epsilon is facultative


# primary capsule
def PrimaryCapsule(inputs, n_primary_caps, dim_primary_caps, conv_kernel_size, padding):
    conv = layers.Conv2D(filters=int(dim_primary_caps * n_primary_caps),
                         kernel_size=conv_kernel_size,
                         strides=2,
                         padding=padding,
                         data_format='channels_first',
                         name='primary_caps')(inputs)  # (None, n_primary_caps*dim_primary_caps, 4, 64)

    outputs = layers.Reshape(target_shape=(-1, dim_primary_caps),
                             name='primary_caps_reshape')(conv)  # (None, n_primary_caps*4*64, dim_primary_caps)
    tf.print('PRIMARY CAPSULE SHAPE', outputs.shape)

    return layers.Lambda(squash, name='primary_caps_squash')(outputs)


# secondary capsule
class SecondaryCapsule(layers.Layer):
    def __init__(self, n_secondary_caps, dim_secondary_caps, r,
                 epsilon,
                 kernel_initializer='glorot_uniform'):
        super(SecondaryCapsule, self).__init__()
        self.n_secondary_caps = n_secondary_caps
        self.dim_secondary_caps = dim_secondary_caps
        self.r = r
        self.epsilon = epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, 'input should have shape (None, n_primary_caps, dim_primary_caps)'
        self.n_primary_caps = input_shape[1]  # this is actually n_primary_caps*4*64
        self.dim_primary_caps = input_shape[2]

        # weight matrix
        self.W = self.add_weight(shape=(self.n_secondary_caps, self.n_primary_caps,
                                        self.dim_secondary_caps, self.dim_primary_caps),
                                 initializer=self.kernel_initializer,
                                 name='transformation_matrix',
                                 trainable=True)
        tf.print('WEIGHT MATRIX SHAPE', self.W.shape)  # (2, 16384, 32, 16)

        # self.built = True
        super(SecondaryCapsule, self).build(input_shape)

    @tf.function
    def call(self, inputs, training=None):
        u = tf.expand_dims(tf.expand_dims(inputs, 1), -1)
        u = tf.tile(u, [1, self.n_secondary_caps, 1, 1, 1])
        # multiply with weight matrix to get estimator of u
        u_hat = tf.squeeze(tf.map_fn(lambda x: tf.matmul(self.W, x), elems=u))  # (None, 2, 16384, 32)

        # routing algorithm
        with tf.name_scope('dynamic_routing'):
            # initialize b as 0
            b_size = tf.cast(tf.size(inputs), dtype=tf.float32) / (self.dim_primary_caps / self.n_secondary_caps)
            b = tf.zeros(tf.cast(b_size, dtype=tf.int32))
            b = tf.reshape(b, (-1, self.n_secondary_caps, 1, self.n_primary_caps))  # (None, 2, 1, 16384)

            for i in range(self.r):
                c = tf.nn.softmax(b, axis=-1)  # (None, n_sec_caps, 1, n_primary_caps)

                s = tf.matmul(c, u_hat)  # (None, n_secondary_caps, 1, dim_secondary_caps)
                v = squash(s, epsilon=self.epsilon)
                # v = tf.nn.sigmoid(s)

                agreement = tf.matmul(v, u_hat, transpose_b=True)

                b += agreement

        v_squeezed = tf.squeeze(v)
        return tf.reshape(v_squeezed, [-1, self.n_secondary_caps, self.dim_secondary_caps])

    def compute_output_shape(self, input_shape):
        return tuple([None, self.n_secondary_caps, self.dim_secondary_caps])

    def get_config(self):  # add params from __init()__ to config to be able to save model
        config = {
            'n_secondary_caps': self.n_secondary_caps,
            'dim_secondary_caps': self.dim_secondary_caps,
            'r': self.r,
            'epsilon': self.epsilon,
            'kernel_initializer': self.kernel_initializer
        }
        base_config = super(SecondaryCapsule, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def bn_relu(layer, nodes, drop_prob):
    norm = layers.BatchNormalization()(layer)
    drop = layers.Dropout(drop_prob)(norm)
    dense = layers.Dense(nodes, activation='relu',
                         kernel_initializer=keras.initializers.HeNormal())(drop)
    return dense


class Masking(layers.Layer):
    def __init__(self, n_secondary_caps, dim_secondary_caps, epsilon):
        super(Masking, self).__init__(name='regression')
        self.n_secondary_caps = n_secondary_caps
        self.epsilon = epsilon
        self.dim_secondary_caps = dim_secondary_caps

    @tf.function
    def call(self, inputs):
        # inputs.shape = (None, n_secondary_caps, dim_secondary_caps)
        x = tf.sqrt(tf.reduce_sum(tf.square(inputs), axis=-1)) + self.epsilon
        # x.shape = (None, n_secondary_caps)
        m = tf.argmax(x, axis=1)
        # m.shape = None
        mask = tf.one_hot(indices=m, depth=self.n_secondary_caps)
        # mask.shape = (None, n_secondary_caps)

        masked = tf.keras.backend.batch_flatten(
            inputs * tf.expand_dims(mask, -1))  # (None, n_caps, dim_caps) * (None, n_caps, 1)
        # masked.shape = (None, n_caps*dim_caps)

        l = tf.keras.backend.batch_flatten(tf.cast(x, dtype=tf.float32) * mask)
        l_sl = tf.slice(l, [0, 1], [-1, 1])

        return tf.reshape(masked, [-1, self.n_secondary_caps * self.dim_secondary_caps], name='masked'), \
               tf.reshape(l_sl, [-1, ], name='regression')

    def get_config(self):  # add params from __init()__ to config to be able to save model
        config = {
            'n_secondary_caps': self.n_secondary_caps,
            'dim_secondary_caps': self.dim_secondary_caps,
            'epsilon': self.epsilon
        }
        base_config = super(Masking, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def CapsNet(n_conv_kernels, conv_kernel_size, padding,
            n_primary_caps, dim_primary_caps,
            n_secondary_caps, dim_secondary_caps,
            r,
            units_1, units_2, drop_prob,
            inp_channels, tok_per_window, embedding_dim,
            epsilon, lr, alpha):
    x = layers.Input(shape=(inp_channels, tok_per_window, embedding_dim),
                     name='input')
    tf.print('INPUT SHAPE', x.shape)

    # initial convolution to extract basic features
    init_conv = layers.Conv2D(filters=n_conv_kernels,
                              kernel_size=conv_kernel_size,
                              strides=1,
                              kernel_initializer=keras.initializers.HeNormal(),
                              activation='relu',
                              padding=padding,
                              data_format='channels_first',
                              name='initial_convolution')(
        x)  # (None, n_conv_kernels, tok_per_window, embedding_dim)
    tf.print('INITIAL CONVOLUTION SHAPE', init_conv.shape)

    # primary capsule to extract lower-level features
    prim_caps = PrimaryCapsule(init_conv,
                               n_primary_caps=n_primary_caps,
                               dim_primary_caps=dim_primary_caps,
                               conv_kernel_size=conv_kernel_size,
                               padding=padding)  # (None, n_primary_caps*4*64, dim_primary_caps)
    tf.print('PRIMARY CAPSULE SHAPE', prim_caps.shape)

    # secondary caps with dynamic routing
    sec_caps = SecondaryCapsule(n_secondary_caps=n_secondary_caps,
                                dim_secondary_caps=dim_secondary_caps,
                                r=r,
                                epsilon=epsilon)(prim_caps)  # (None, n_secondary_caps, dim_secondary_caps)
    tf.print('SECONDARY CAPSULE SHAPE', sec_caps.shape)

    # mask
    masked, pred_count = Masking(n_secondary_caps=n_secondary_caps,
                                 dim_secondary_caps=dim_secondary_caps,
                                 epsilon=epsilon)(sec_caps)

    # decoder
    dense1 = bn_relu(masked, units_1, drop_prob=drop_prob)
    dense2 = bn_relu(dense1, units_2, drop_prob=drop_prob)
    dense3 = bn_relu(dense2, int(inp_channels * tok_per_window * embedding_dim), drop_prob=drop_prob)

    dec = layers.Reshape(target_shape=(inp_channels, tok_per_window, embedding_dim),
                         name='reconstruction')(dense3)
    tf.print('RECONCTRUCTION SHAPE', dec.shape)

    # regression
    tf.print('REGRESSION SHAPE', pred_count.shape)

    # concatenate to model
    model = keras.Model(inputs=[x], outputs=[dec, pred_count])

    # loss functions
    # MSE for both reconstruction and regression
    losses = {'reconstruction': 'mean_absolute_error',
              'regression': 'mean_squared_error'}

    loss_weights = {'reconstruction': alpha,
                    'regression': 1.0}

    metrics = {'reconstruction': ['mean_squared_error'],
               'regression': ['mean_absolute_error']}

    # compile model
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    opt = hvd.DistributedOptimizer(opt)

    model.compile(optimizer=opt,
                  loss=losses,
                  loss_weights=loss_weights,
                  metrics=metrics,
                  experimental_run_tf_function=False)
    return model


# @use_named_args(dimensions=dimensions)

def fitness(batch_size, max_lr, n_conv_kernels, conv_kernel_size, n_primary_caps, dim_primary_caps, r,
            units_1, units_2, drop_prob):

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    model = CapsNet(n_conv_kernels=n_conv_kernels, conv_kernel_size=conv_kernel_size, padding='same',
                    n_primary_caps=n_primary_caps, dim_primary_caps=dim_primary_caps,
                    n_secondary_caps=2, dim_secondary_caps=16, r=r, units_1=units_1, units_2=units_2,
                    drop_prob=drop_prob, inp_channels=2, tok_per_window=tokPerWindow, embedding_dim=embeddingDim,
                    epsilon=1e-07, lr=max_lr * hvd.size(), alpha=1e-03)

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

    print('train for {}, validate for {} steps per epoch'.format(steps, val_steps))

    fit = model.fit(x=emb,
                    y=[emb, counts],
                    batch_size=batchSize,
                    validation_data=(emb_test,
                                     [emb_test, counts_test]),
                    validation_batch_size=batchSize,
                    steps_per_epoch=steps,
                    validation_steps=val_steps,
                    epochs=epochs,
                    callbacks=callbacks,
                    initial_epoch=1,
                    max_queue_size=1,
                    verbose=2 if hvd.rank() == 0 else 0,
                    shuffle=True)

    loss = min(fit.history['val_regression_loss'])
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
        print('VALIDATION REGRESSION LOSS: ', loss)
        print('#######################################################################################################')
        print('#######################################################################################################')

        pd.DataFrame.to_csv(m, '/scratch2/hroetsc/Hotspots/results/opt_metrics_{}.txt'.format(loss_str), header=False, index=False)

        # merge actual and predicted counts
        prediction = pd.DataFrame({"Accession": tokens_test[:, 0],
                                   "window": tokens_test[:, 1],
                                   "label": labels_test,
                                   "count": counts_test,
                                   "pred_count": pred[1].flatten()})

        pd.DataFrame.to_csv(prediction,
                            '/scratch2/hroetsc/Hotspots/results/opt_prediction_{}.csv'.format(loss_str),
                            index=False)

        del model

        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()

    return loss


lr = [1e-07, 5e-07, 1e-06, 5e-06, 1e-05, 5e-05, 1e-04, 5e-04, 1e-03, 5e-03, 1e-02, 5e-02]
r = [2, 3, 4, 5]

for i in lr:
    for j in r:
        fitness(batch_size=16, max_lr=i,
                n_conv_kernels=256, conv_kernel_size=3,
                n_primary_caps=32, dim_primary_caps=16,
                r=j,
                units_1=512,
                units_2=1024,
                drop_prob=0.5)

# gbrt = gbrt_minimize(func=fitness,
#                      dimensions=dimensions,
#                      n_calls=no_opt_iterations,
#                      x0=default_parameters)
#
# if hvd.rank() == 0:
#     gbrt_results = list(zip(gbrt.func_vals, gbrt.x_iters))
#     pd.DataFrame.to_csv(pd.DataFrame(gbrt_results), '/scratch2/hroetsc/Hotspots/results/opt_results.csv')

tf.keras.backend.clear_session()
