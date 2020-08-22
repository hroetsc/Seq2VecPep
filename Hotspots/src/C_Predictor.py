### HEADER ###
# HOTSPOT PREDICTION
# description: set up neural network to predict hotspot density along the human proteome
# input: table with tokens in sliding windows and scores, table with tf-idf weighted embeddings
# output: model
# author: HR


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

tf.keras.backend.clear_session()

### initialize GPU training environment

# ...

### INPUT ###
tokensAndCounts = pd.read_csv('results/windowTokens.csv')
weights = pd.read_csv('results/tfidf-weights.csv')

### HYPERPARAMETERS ###
embeddingDim = 100
tokPerWindow = 8

epochs = 10
batchSize = 64

### MAIN PART ###
### format input
# tmp!
idx = np.random.randint(0, tokensAndCounts.shape[0], int(1e03))

tokens = np.array(tokensAndCounts.loc[idx, ['Accession', 'tokens']], dtype='object')
counts = np.array(tokensAndCounts.loc[idx, ['counts']], dtype='float64')


#### batch generator
# generates sequence representation on the fly

def findEmbeddings(words, accession):

    emb = np.empty((len(words), embeddingDim), dtype='float64')

    for w, a in words, accession:
        res = weights[(weights['Accession'] == a) & (weights['token'] == w)].iloc[:, 3:(embeddingDim+3)]
        emb = np.append(emb, np.array(res), axis=1)

    return emb

def sequenceGenerator(tokens):

    # split tokens
    tokens_split = [str.split(' ') for str in cnt for cnt in list(tokens[:, 1])]
    acc = list(tokens[:, 0])

    # find them in the big weights file
    embMatrices = [findEmbeddings(words, accession) for words, accession in [tokens_split, acc]]

    # convert list into np.array of matrices
    embMatrices = np.array(embMatrices, dtype='float64')

    return embMatrices


### build and compile model
def build_and_compile_model():

    ## layers
    inp = keras.Input(shape=(tokPerWindow, embeddingDim))

    # first convolution
    conv1 = layers.Conv2D(32, kernel_size=(2,2), input_shape=(tokPerWindow, embeddingDim),
                          strides=(2,2),
                          activation='relu')(inp)
    pool1 = layers.MaxPool2D((2,2))(conv1)

    # second convolution
    conv2 = layers.Conv2D(64, kernel_size=(2,2), input_shape=(tokPerWindow, embeddingDim),
                          strides=(1,1),
                          activation='relu')(pool1)
    pool2 = layers.MaxPool2D((2,2))(conv2)

    # flatten
    flat = layers.Flatten()(pool2)

    # fully connected layer
    fully = layers.Dense(128, activation='relu')(flat)

    # dropout and output layer
    drop = layers.Dropout(0.2)(fully)
    out = layers.Dense(1, activation='sigmoid')(drop)

    ## model
    model = keras.Model(inputs=inp, outputs=out)

    ## compile model
    opt = keras.optimizers.Adam(learning_rate=0.05)
    model.compile(loss=keras.losses.MeanSquaredError(),
                  optimizer=opt,
                  metrics=['squared_hinge', 'categorical_hinge', 'categorical_crossentropy', 'accuracy'],
                  experimental_run_tf_function=False)

    return model


#### train model
# split data
tokens_train, tokens_test, counts_train, counts_test = train_test_split(tokens, counts, test_size=.2)

# define callbacks - adapt later for multi-node training
# early stopping if model is already converged
es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                        mode = 'min',
                                        patience = 5,
                                        min_delta = 0.0005,
                                        verbose = 1)

# define number of steps
steps = int(np.ceil(counts_train.shape[0] / batchSize))
val_steps = int(np.ceil(counts_test.shape[0] / batchSize))

# fit model
model = build_and_compile_model()
model.fit(x=sequenceGenerator(tokens_train),
          y=counts_train,
          validation_data=[sequenceGenerator(tokens_test), counts_test],
          batch_size=batchSize,
          steps_per_epoch=steps,
          validation_steps=val_steps,
          epochs=epochs,
          callbacks=[es],
          initial_epoch=0,
          verbose=1,
          shuffle=True)


### OUTPUT ###
# save weights
model.save_weights('results/weights.h5')
#save entire model
model.save('results/model.h5')

# save metrics
val = []
name = list(fit.history.keys())
for i, elem in enumerate(fit.history.keys()):
    val.append(fit.history[elem])

m = list(zip(name, val))
m = pd.DataFrame(m)
pd.DataFrame.to_csv(m, 'results/metrics.txt', header=False, index = False)