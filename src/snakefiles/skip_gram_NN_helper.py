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

def generate_skipgrams(wid, windowSize, vocab_size):
    skip_grams = skipgrams(sequence = wid,
                                vocabulary_size = vocab_size + 1,
                                window_size = windowSize,
                                categorical = False,
                                shuffle = False,
                                negative_samples = 1)
    gc.collect()
    return skip_grams


def parallel_processing(n, wid, windowSize, vocab_size, n_batches, file):
    #print("generating batch {} of {}".format(n+1, n_batches))

    skip_grams = generate_skipgrams(wid, windowSize, vocab_size)
    pairs, labels = skip_grams

    # save everything in one file
    f = open(file, 'ab').close() # overwrite existing files
    f = open(file, 'ab')
    np.savetxt(f, np.array(list(zip(*pairs))[0], list(zip(*pairs))[1], labels, dtype='int32'))
    f.close()

    gc.collect()
