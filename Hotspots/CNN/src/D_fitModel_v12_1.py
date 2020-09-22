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

epochs = 400
batchSize = 32
pseudocounts = 1
no_cycles = 4
no_features = int(tokPerWindow * 10)
augment = False

epochs = int(np.ceil(epochs / hvd.size()))
batchSize = batchSize * hvd.size()

print('number of epochs, adjusted by number of GPUs: ', epochs)
print('batch size, adjusted by number of GPUs: ', batchSize)
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
# write model
class CapsNet(keras.Model):
    def __init__(self,
                 n_conv_kernels, conv_kernel_size, padding,
                 n_primary_caps, primary_caps_vector,
                 n_secondary_caps, secondary_caps_vector,
                 r,
                 units_1, units_2,
                 inp_channels, tok_per_window, embedding_dim,
                 epsilon, m_plus, m_minus, lamb, alpha,
                 name='capsnet'):
        super(CapsNet, self).__init__(name=name)

        self.n_conv_kernels = n_conv_kernels
        self.conv_kernel_size = conv_kernel_size
        self.padding = padding

        self.n_primary_caps = n_primary_caps
        self.primary_caps_vector = primary_caps_vector
        self.n_secondary_caps = n_secondary_caps
        self.secondary_caps_vector = secondary_caps_vector

        self.r = r

        self.units_1 = units_1
        self.units_2 = units_2
        self.inp_channels = inp_channels
        self.tok_per_window = tok_per_window
        self.embedding_dim = embedding_dim
        self.output_dim = int(self.tok_per_window * self.embedding_dim * self.inp_channels)

        # does only work with padding = "same"!
        self.vector_dim = int(self.tok_per_window * 0.5 * self.embedding_dim * 0.5 * self.n_primary_caps)
        # e.g. 16384

        self.epsilon = epsilon
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lamb = lamb
        self.alpha = alpha

        with tf.name_scope('variables'):
            # initial convolution detects basic features
            self.convolution = layers.Conv2D(filters=self.n_conv_kernels,
                                             kernel_size=self.conv_kernel_size,
                                             strides=1,
                                             kernel_initializer=keras.initializers.HeNormal(),
                                             activation='relu',
                                             padding=self.padding,
                                             data_format='channels_first',
                                             name='convolution_layer')

            # primary capsule should detect low-level features from basic features that were detected by initial convolution
            self.primary_capsule = layers.Conv2D(filters=int(self.n_primary_caps * self.primary_caps_vector),
                                                 kernel_size=self.conv_kernel_size,
                                                 strides=2,
                                                 padding=self.padding,
                                                 data_format='channels_first',
                                                 name='primary_capsule')

            # affine transform weight matrix
            self.w = tf.Variable(tf.random_normal_initializer()(shape=(1,
                                                                       self.vector_dim,
                                                                       self.n_secondary_caps,
                                                                       self.secondary_caps_vector,
                                                                       self.primary_caps_vector)),
                                 dtype=tf.float32,
                                 name="pose_estimation", trainable=True)

            # dense layers (decoder)
            self.dense_1 = layers.Dense(self.units_1, activation='relu',
                                        kernel_initializer=keras.initializers.HeNormal())
            self.dense_2 = layers.Dense(self.units_2, activation='relu',
                                        kernel_initializer=keras.initializers.HeNormal())
            self.dense_3 = layers.Dense(self.output_dim, activation='relu',
                                        kernel_initializer=keras.initializers.HeNormal())

    def build(self, input_shape):
        pass

    # non-linear activation: squash function
    def squash(self, s):
        with tf.name_scope('squash_function') as scope:
            s_norm = tf.norm(s, axis=-1, keepdims=True)
            return (tf.square(s_norm) / (1 + tf.square(s_norm))) * (s / (s_norm + self.epsilon))  # epsilon is facultative

    # loss
    # make sure that norm does not become 0
    def safe_norm(self, v, axis=-1):
        v_ = tf.reduce_sum(tf.square(v), axis=axis, keepdims=True)
        return tf.sqrt(v_ + self.epsilon)

    # loss function combines margin loss and reconstruction loss
    def loss_function(self, v, dec, y, x):
        prediction = self.safe_norm(v)
        prediction = tf.reshape(prediction, shape=(-1, self.n_secondary_caps))

        # margin loss
        left_margin = tf.square(tf.maximum(0.0, self.m_plus - prediction))
        right_margin = tf.square(tf.maximum(0.0, prediction - self.m_minus))

        l = tf.add(y * left_margin, self.lamb * (1.0 - y) * right_margin)
        margin_loss = tf.reduce_mean(tf.reduce_sum(l, axis=-1))

        # reconstruction (decoding) loss
        y_matrix_flat = tf.reshape(x, shape=(-1, self.output_dim))
        reconstruction_loss = tf.reduce_mean(tf.square(y_matrix_flat - dec))

        loss = tf.add(margin_loss, self.alpha * reconstruction_loss)
        return loss

    @tf.function
    def call(self, inputs, training=True):
        x_inp, y = inputs
        # input x: (None, 2, 8, 128)
        # input y: (None, 2)

        # primary convolution
        x = self.convolution(x_inp)  # (None, n_conv_kernels, 8, 128)
        # lower-level capsule, returns vector
        x = self.primary_capsule(x)  # (None, n_primary_caps*primary_caps_vector=1024, 4, 64)

        # secondary capsule detects high-level features
        with tf.name_scope('form_capsule'):
            # reshape vector output of lower-level capsule
            u = tf.reshape(x, shape=(-1,
                                     self.n_primary_caps * x.shape[2] * x.shape[3],
                                     # channels first! should be equal to vector_dim
                                     self.primary_caps_vector))  # (None, 16384, 16) e.g.
            u = tf.expand_dims(u, axis=-2)
            u = tf.expand_dims(u, axis=-1)  # (None, 16384, 1, 16, 1)
            # multiply with weight matrix to get estimator of u
            u_hat = tf.matmul(self.w, u)
            u_hat = tf.squeeze(u_hat, [4])  # (None, 16384, 2, 32)

        # determine scalar weights by applying dynamic routing algorithm
        with tf.name_scope('dynamic_routing'):
            # initialize parameter b
            b = tf.zeros(shape=(None,
                                self.vector_dim,
                                self.n_secondary_caps,
                                1))  # (None, 16384, 2, 1)

            # iterate to refine b
            for i in range(self.r):
                # softmax ensures probability characteristics (non-negative, sum up to 1)
                c = tf.nn.softmax(b, axis=-2)  # (None, 16834, 2, 1)
                # scale down all input vectors and add them up
                s = tf.reduce_sum(tf.multiply(c, u_hat), axis=1, keepdims=True)  # (None, 1, 2, 32)
                v = self.squash(s)

                # update b
                agreement = tf.squeeze(tf.matmul(tf.expand_dims(u_hat, axis=-1),
                                                 tf.expand_dims(v, axis=-1),
                                                 transpose_a=True),
                                       [4])  # (None, 16384, 2, 1)
                b += agreement

        # mask labels
        with tf.name_scope('masking'):
            y_exp = tf.expand_dims(y, axis=-1)
            y_exp = tf.expand_dims(y_exp, axis=1)
            mask = tf.cast(y_exp, dtype=tf.float32)  # (None, 1, 2, 1)
            v_masked = tf.multiply(mask, v)  # (None, 1, 2, 32)

        with tf.name_scope('decoding'):
            v_ = tf.reshape(v_masked, shape=(-1, self.n_secondary_caps * self.secondary_caps_vector))  # (None, 64)
            dec = self.dense_1(v_)
            dec = self.dense_2(dec)
            dec = self.dense_3(dec)  # (None, 1024)

        loss = self.loss_function(v, dec, y, x_inp)
        self.add_loss(loss)

        # # 2 outputs: secondary capsule output (squashed) and decoded sequence
        return v, dec



## compile model
params = {'n_conv_kernels': 512,
          'conv_kernel_size': (3, 3),
          'padding': 'same',
          'n_primary_caps': 64,
          'primary_caps_vector': 16,
          'n_secondary_caps': 2,
          'secondary_caps_vector': 32,
          'r': 3,
          'units_1': 1024,
          'units_2': 2048,
          'inp_channels': 2,
          'tok_per_window': int(tokPerWindow),
          'embedding_dim': int(embeddingDim),
          'epsilon': 1e-07,
          'm_plus': 0.9,
          'm_minus': 0.1,
          'lamb': 0.5,
          'alpha': 5e-03}

model = CapsNet(**params)

lr = 0.001 * hvd.size()
tf.print('learning rate, adjusted by number of GPUS: ', lr)
opt = tf.keras.optimizers.Adam(learning_rate=lr)
opt = hvd.DistributedOptimizer(opt)

model.compile(optimizer=opt,
              metrics=['mean_absolute_error', 'mean_absolute_percentage_error', 'accuracy'],
              experimental_run_tf_function=False)

tf.print('......................................................')
tf.print('CAPSULE NEURAL NETWORK')
tf.print('optimizer: Adam')
tf.print("learning rate: --> cosine annealing")
tf.print(params.keys())
tf.print(params.values())
tf.print('......................................................')

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

if hvd.rank() == 0:
    model.summary()
    print('train for {}, validate for {} steps per epoch'.format(steps, val_steps))
    print('using sequence generator')

# one-hot encoded labels!
counts_1h = tf.one_hot(counts, depth=2)
counts_test_1h = tf.one_hot(counts_test, depth=2)

fit = model.fit(x=[emb, counts_1h],
                y=[counts_1h, emb],
                batch_size=batchSize,
                validation_data=([emb_test, counts_test_1h],
                                 [counts_test_1h, emb_test]),
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
pred = model.predict(x=[emb_test, counts_test_1h],
                     batch_size=batchSize,
                     verbose=1 if hvd.rank() == 0 else 0,
                     max_queue_size=256)
print('counts:')
print(counts_test)

print('prediction:')
pred_counts = np.array(pred[0].flatten())[:, 0]
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

tf.keras.backend.clear_session()
