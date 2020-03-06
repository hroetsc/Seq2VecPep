### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  convert tokens to numerical vectors using a skip-gram neural network
# input:        tokens generated by BPE algorithm in antigen_segmentation.R
# output:       embedded tokens (weights and tehir IDs)
# author:       HR

import os
import re
import gc
import numpy as np
import pandas as pd

# HYPERPARAMETERS
windowSize = 20
embeddingDim = 200

#epochs = 500
epochs = 10 # for testing
workers = 11

batchSize = 256
valSplit = 0.20

# INPUT
os.chdir('/home/hroetsc/Documents/ProtTransEmbedding/Snakemake/results/encoded_proteome')
tokens = open('words_as_text.txt', 'r')
tokens = tokens.read()
# for testing
tokens = tokens[38000:42000]
# convert tokens into list
tokens_list = tokens.split(" ")

# TEXT PREPROCESSING
# https://www.kdnuggets.com/2018/04/implementing-deep-learning-methods-feature-engineering-text-data-skip-gram.html
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import text
from keras.preprocessing.text import *
# generate context pairs
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tokens_list)
word2id = tokenizer.word_index
id2word = {v:k for k, v in word2id.items()}
vocab_size = len(word2id)+1 # bc otherwise model doesn't work
# convert words to IDs
wids = [word2id[w] for w in text_to_word_sequence(tokens)]
print('Vocabulary Size:', vocab_size)
print('Vocabulary Sample:', list(word2id.items())[:10])
# generate skip-grams
skip_grams = skipgrams(wids, vocab_size, window_size=windowSize)
pairs, labels = skip_grams
print("skip grams: (target, context/random) -> label")
for i in range(20): #Visualizing the result (first 20)
    print("({:s} , {:s} ) -> {:d}".format(
          id2word[pairs[i][0]],
          id2word[pairs[i][1]],
          labels[i]))

# get subword information
#from keras.preprocessing.text import one_hot
#from keras.preprocessing.sequence import pad_sequences

#def get_subword(subword):
#    for i in range(1, len(subword)):
#        for c in get_subword(subword[i:]):
#            yield (subword[:i],) + c

#subwords = [get_subword(text_to_word_sequence(tokens)[w]) for w in text_to_word_sequence(tokens)]
# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
#padded_subwords = pad_sequences(encoded_subwords, maxlen=4, padding='post', dtype='int32', truncating='post')

# MODEL CREATION
print("model:")
import keras
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

# GPU settings - https://michaelblogscode.wordpress.com/2017/10/10/reducing-and-profiling-gpu-memory-usage-in-keras-with-tensorflow-backend/
# tensorflow wizardy
config = tf.ConfigProto()
config.gpu_options.allow_growth = True # do not pre-allocate memory
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # only allow half of the memory to be allocated
K.tensorflow_backend.set_session(tf.Session(config=config)) # create session

# model - https://adventuresinmachinelearning.com/word2vec-keras-tutorial/
input_target = Input(shape=(1,), dtype='int32', name='target_word')
input_context = Input(shape=(1,), dtype='int32', name='context_word')
# embed input layers
embedding = Embedding(vocab_size, embeddingDim, input_length=1, name='embedding')
# apply embedding
target = embedding(input_target)
target = Reshape((embeddingDim,1))(target)
context = embedding(input_context)
context = Reshape((embeddingDim,1))(context)
# cosine similarity
similarity = dot([target, context], axes=1, normalize=True, name= 'cosine_similarity')
# dot product similarity
dot_product = dot([target, context], axes=1, normalize=False, name = 'dot_product')
dot_product = Reshape((1,))(dot_product)
# add the sigmoid output layer
output = Dense(1, activation='sigmoid', name='1st_sigmoid_layer')(dot_product)
output = Dense(1, activation='sigmoid', name='2nd_sigmoid_layer')(output)
# create the primary training model
model = Model(inputs=[input_target, input_context], outputs=output)
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy']) # binary for binary decisions, categorical for classifications
# create validation model
validation_model = Model(inputs=[input_target, input_context], outputs=similarity)
# view model summary - from keras documentation
print(model.summary())

# TRAINING
print("training the model:")
target_word = np.array(list(zip(*pairs))[0], dtype='int32')
context_word = np.array(list(zip(*pairs))[1], dtype='int32')
Y = np.array(labels, dtype='int32')

# for testing
np.savetxt('tmp_target_word.txt', target_word)
np.savetxt('tmp_context_word.txt', context_word)
np.savetxt('tmp_label.txt', Y)

# split data into training and validation
from sklearn.model_selection import train_test_split
target_train, target_test, context_train, context_test, Y_train, Y_test = train_test_split(target_word, context_word, Y, test_size=valSplit)

# train on batch - generate batches
def batch_generator(target, context, Y, batch_size):
    n_batches = int(np.ceil(target.shape[0]/int(batch_size))) # divide input length by batch size
    counter = 0
    while 1:
        target_batch = np.array(target[batch_size*counter:batch_size*(counter+1)], dtype='int32')
        context_batch = np.array(context[batch_size*counter:batch_size*(counter+1)], dtype='int32')
        Y_batch = np.array(Y[batch_size*counter:batch_size*(counter+1)], dtype='int32')

        # for testing
        #if counter == 4:
        #    np.savetxt('tmp_batch_target.txt', target_batch)
        #    np.savetxt('tmp_batch_context.txt', context_batch)
        #    np.savetxt('tmp_batch_label.txt', Y_batch)

        counter += 1
        yield([target_batch, context_batch], Y_batch)

        if counter >= n_batches: # clean for next epoch
            counter = 0
    gc.collect()


train_generator = batch_generator(target_train, context_train, Y_train, batchSize)
test_generator = batch_generator(target_test, context_test, Y_test, batchSize)

# fit model
steps = np.ceil((vocab_size/batchSize))
val_steps = np.ceil(valSplit*batchSize)

model.fit_generator(generator=train_generator,
                            validation_data=test_generator,
                            steps_per_epoch = steps,
                            validation_steps = val_steps,
                            epochs=epochs,
                            initial_epoch=0,
                            verbose=2,
                            workers=workers,
                            use_multiprocessing=True,
                            shuffle=True)

# OUTPUT
# get word embedding
#weights = model.layers[2].get_weights()[0] # weights of the embedding layer
# save weights of embedding matrix
#df = pd.DataFrame(weights)
#pd.DataFrame.to_csv(df, snakemake.output['weights'], header=False)
#df.head()
# save corresponding IDs
#ids = pd.DataFrame(id2word.values())
#pd.DataFrame.to_csv(ids, snakemake.output['ids'], header=False)
# save model
#model.save(snakemake.output['model'])

# for testing
for l in range(len(model.layers)):
    print(l)
    print(model.layers[l])
    print(model.layers[l].name)
    print(model.layers[l].get_output_at(0))
