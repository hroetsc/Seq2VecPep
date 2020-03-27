### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  convert tokens to numerical vectors using a skip-gram neural network
# input:        tokens generated by BPE algorithm in generate_tokens
# output:       word pairs (target and context word)
# author:       HR

print("### PROTEOME EMBEDDING USING SKIP-GRAM NEURAL NETWORK - 1 ###")
import os
import re
import gc
import numpy as np
import pandas as pd

import threading
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool

import keras
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import text
from keras.preprocessing.text import *

from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Reshape
from keras.layers.embeddings import Embedding
from keras.layers import Dot, concatenate, merge, dot
from keras.layers import *
from keras.engine import input_layer
import tensorflow as tf
from tensorflow.keras import layers
from keras import backend as K

from sklearn.model_selection import train_test_split

# import helper scripts
import skip_gram_NN_helper

gc.enable()

# =============================================================================
# # HYPERPARAMETERS
# =============================================================================
workers = 15
windowSize = 10
keep = 0.3

# INPUT
os.chdir('/home/hanna/Documents/QuantSysBios/ProtTransEmbedding/Snakemake')
words = pd.read_csv('results/encoded_proteome/words.csv', header = 0)

# =============================================================================
# # tmp!!!
# os.chdir('/home/hroetsc/Documents/ProtTransEmbedding/Snakemake')
# words = pd.read_csv('results/encoded_proteome/words.csv', header = 0)
# idx = np.random.randint(0, len(pd.DataFrame.count(words, 1)), size = 10000)
# words = words.iloc[list(idx),]
#
# =============================================================================


# =============================================================================
# # TEXT PREPROCESSING
# =============================================================================
print("TEXT PREPROCESSING")
# extract tokens
tokens = list(words['tokens'])
tokens = ' '.join(tokens)
tokens = tokens.split(' ')

# https://www.kdnuggets.com/2018/04/implementing-deep-learning-methods-feature-engineering-text-data-skip-gram.html
# functions needed to generate tokens
tokenizer = text.Tokenizer(num_words = len(tokens))
tokenizer.fit_on_texts(tokens)
word2id = tokenizer.word_index
id2word = {v:k for k, v in word2id.items()}

# generate skip-grams for every protein independently
# word-IDs have to be generated for whole proteome to prevent duplications
print('CONVERT WORDS TO IDs')
wids = [[word2id[w] for w in text.text_to_word_sequence(token)] for token in words['tokens']]
no_of_proteins = len(wids)
print("number of proteins / batches: {}".format(no_of_proteins))
vocab_size = len(word2id.keys())
print("vocabulary size: {}".format(vocab_size))

print('GENERATE SKIP-GRAMS')
n_batches = no_of_proteins
print("skip-grams are calculated in batches (equivalent to proteins)")

# parallel processing
pool = multiprocessing.Pool(workers)

if __name__ == "__main__":
    pool.starmap( skip_gram_NN_helper.parallel_processing,
                ([[n, wids[n], windowSize, vocab_size, n_batches, 'results/embedded_proteome/skipgrams.txt', keep] for n in range(n_batches)]) )
print('done with generating skip-grams')

# =============================================================================
# # OUTPUT
# =============================================================================
print("SAVE OUTPUT")

# save corresponding IDs
ids = pd.DataFrame(word2id.items())
pd.DataFrame.to_csv(ids, 'results/embedded_proteome/seq2vec_ids.csv', header=False, index = False)

#pool.close() # to save memory
#pool.join()

print("### PROTEOME EMBEDDING USING SKIP-GRAM NEURAL NETWORK - 2 ###")

import os
import gc
import threading
import numpy as np
import pandas as pd

import keras

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Reshape
from keras.layers.embeddings import Embedding
from keras.layers import Dot, concatenate, merge, dot
from keras.layers import *
from keras.engine import input_layer
import tensorflow as tf
from tensorflow.keras import layers
from keras import backend as K

from keras.utils import Sequence
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

gc.enable()

# GPU settings - https://michaelblogscode.wordpress.com/2017/10/10/reducing-and-profiling-gpu-memory-usage-in-keras-with-tensorflow-backend/
# tensorflow wizardy
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True # do not pre-allocate memory
# config.gpu_options.per_process_gpu_memory_fraction = 0.5 # only allow half of the memory to be allocated
# K.tensorflow_backend.set_session(tf.Session(config=config)) # create session

# =============================================================================
# # HYPERPARAMETERS
# =============================================================================
workers = 12

# window of a word: [i - window_size, i + window_size+1]
embeddingDim = 100
epochs = 5 # baaaaaaah #200 min

batchSize = 32
valSplit = 0.20

learning_rate = 0.01
adam_decay = 0.005200110247661778

# =============================================================================
# # INPUT
# =============================================================================
print("LOAD DATA")
skip_grams = pd.read_csv('results/embedded_proteome/skipgrams.txt', sep = " ", header = None)
ids = pd.read_csv('results/embedded_proteome/seq2vec_ids.csv', header = None)

# split skip-grams into target, context and label np.array()
target_word = np.array(skip_grams.iloc[:,0], dtype = 'int32')
context_word = np.array(skip_grams.iloc[:,1], dtype = 'int32')
Y = np.array(skip_grams.iloc[:,2], dtype = 'int32')

print('target word vector')
target_word = target_word.reshape(target_word.shape[0],)
print(target_word)

print('context word vector')
context_word = context_word.reshape(context_word.shape[0],)
print(context_word)

print('label vector')
Y = Y.reshape(Y.shape[0],)
print(Y)

vocab_size = len(ids.index)+2
print("vocabulary size (number of target word IDs +2): {}".format(vocab_size))

# =============================================================================
# # MODEL CREATION
# =============================================================================
print("MODEL GENERATION")

# model - https://adventuresinmachinelearning.com/word2vec-keras-tutorial/
# https://adventuresinmachinelearning.com/word2vec-tutorial-tensorflow/
input_target = keras.layers.Input(((1,)), name='target_word')
input_context = keras.layers.Input(((1,)), name='context_word')

# embed input layers
# https://github.com/keras-team/keras/issues/3110
embedding = Embedding(input_dim = vocab_size,
                        output_dim = embeddingDim,
                        input_length = 1,
                        embeddings_initializer = 'glorot_uniform',
                        name = 'embedding')
# later create lookup table with weights so that one could initialize the embedding layer with pretrained weights

# apply embedding
target = embedding(input_target)
target = Reshape((embeddingDim,1), name='target_embedding')(target) # every individual skip-gram has dimension embedding x 1
context = embedding(input_context)
context = Reshape((embeddingDim,1), name='context_embedding')(context)

# dot product similarity - normalize to get value between 0 and 1!
dot_product = dot([target, context], axes = 1, normalize = True, name = 'dot_product')
dot_product = Reshape((1,))(dot_product)

# add the sigmoid dense layer

output = Dense(1, activation = 'sigmoid', kernel_initializer = 'glorot_uniform', name='1st_dense')(dot_product)
output = Dropout(0.5)(output)
output = Dense(1, activation = 'sigmoid', kernel_initializer = 'glorot_uniform', name='5th_dense')(output)

# create the primary training model
model = Model(inputs=[input_target, input_context], outputs=output)

#adam = Adam(lr=learning_rate, decay=adam_decay)
#model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy']) # binary for binary decisions, categorical for classifications

# binary classification loss functions
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='squared_hinge', optimizer='adam', metrics=['accuracy'])

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

# USE FIT_GENERATOR
# train on batch - make batch generator threadsafe (with small number of steps and multiprocessing otherwise duplicated batches occur)
# https://stackoverflow.com/questions/56441216/on-fit-generator-and-thread-safety

# train on batch - generate batches
# https://stackoverflow.com/questions/46493419/use-a-generator-for-keras-model-fit-generator
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

# make batch generator suitable for multiprocessing - use keras.utils.Sequence class
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence


# https://keunwoochoi.wordpress.com/2017/08/24/tip-fit_generator-in-keras-how-to-parallelise-correctly/
# =============================================================================
# class threadsafe_iter:
#     """Takes an iterator/generator and makes it thread-safe by
#     serializing call to the `next` method of given iterator/generator.
#     """
#     def __init__(self, it):
#         self.it = it
#         self.lock = threading.Lock()
#
#     def __iter__(self):
#         return self
#
#     def next(self):
#         with self.lock:
#             return self.it.next()
#
# def threadsafe_generator(f):
#     """A decorator that takes a generator function and makes it thread-safe.
#     """
#     def g(*a, **kw):
#         return threadsafe_iter(f(*a, **kw))
#     return g
#
# =============================================================================


# =============================================================================
# @threadsafe_generator
# def batch_generator(target, context, Y, batch_size):
#     n_batches = int(np.ceil(target.shape[0]/int(batch_size))) # divide input length by batch size
#     counter = 0
#     #threading.Lock()
#     while 1:
#         target_batch = target[batch_size*counter:batch_size*(counter+1)]
#         context_batch = context[batch_size*counter:batch_size*(counter+1)]
#         Y_batch = Y[batch_size*counter:batch_size*(counter+1)]
#
#         #print([target_batch, context_batch], Y_batch)
#
#         print(counter)
#
#         counter += 1
#         yield([target_batch, context_batch], Y_batch)
#
#         if counter >= n_batches: # clean for next epoch
#             counter = 0
#
#         gc.collect()
# =============================================================================

class BatchGenerator(keras.utils.Sequence):

     def __init__(self, target, context, Y, batch_size):
         self.target, self.context, self.Y = target, context, Y
         self.batch_size = batch_size

     def __len__(self):
         return int(np.ceil(len(self.target) / float(self.batch_size)))

     def __getitem__(self, idx):

         #print(idx)

         batch_target = self.target[idx*self.batch_size : (idx + 1)*self.batch_size]
         batch_context = self.context[idx*self.batch_size : (idx + 1)*self.batch_size]
         batch_Y = self.Y[idx*self.batch_size : (idx + 1)*self.batch_size]

         return [batch_target, batch_context], batch_Y

     def on_epoch_end(self):
         pass

# apply batch generator
print("generating batches for model training")
train_generator = BatchGenerator(target_train, context_train, Y_train, batchSize)
test_generator = BatchGenerator(target_test, context_test, Y_test, batchSize)

# fit model
print("fit the model")

# can be ignored in case batch generator uses keras.utils.Sequence() class
#steps = np.ceil(target_train.shape[0]/batchSize)
#val_steps = np.ceil(target_test.shape[0]/batchSize)

fit = model.fit_generator(generator = train_generator,
                    validation_data = test_generator,
                    epochs = epochs,
                    initial_epoch = 0,
                    verbose=2,
                    max_queue_size = 1,
                    workers = workers,
                    use_multiprocessing = False,
                    shuffle = False)

# shuffle has to be false bc BatchBenerator can't cope with shuffled data!

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
pd.DataFrame.to_csv(df, 'results/embedded_proteome/seq2vec_weights.csv', header=False)
df.head()

# save model
model.save('results/embedded_proteome/model.h5')

# save accuracy and loss
m = open('results/embedded_proteome/model_metrics.txt', 'w')
m.write("accuracy \t {} \n val_accuracy \t {} \n loss \t {} \n val_loss \t {}".format(fit.history['accuracy'], fit.history['val_accuracy'], fit.history['loss'], fit.history['val_loss']))
m.close()

K.clear_session()
tf.reset_default_graph()
