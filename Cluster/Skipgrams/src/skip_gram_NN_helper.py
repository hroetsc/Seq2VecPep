print("### PROTEOME EMBEDDING USING SKIP-GRAM NEURAL NETWORK - HELPER SCRIPT ###")
import os
import sys
import re
import gc
import numpy as np
import pandas as pd

import argparse
import threading
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool

import keras
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import text
from keras.preprocessing.text import *

def generate_skipgrams(wid, windowSize, vocab_size, negSkipgrams):
    skip_grams = skipgrams(sequence = wid,
                                vocabulary_size = vocab_size + 1,
                                window_size = windowSize,
                                categorical = False,
                                shuffle = False,
                                negative_samples = negSkipgrams)
    gc.collect()
    return skip_grams


def parallel_processing(n, wid, windowSize, vocab_size, n_batches, file, keep, negSkipgrams):
    #print("generating batch {} of {}".format(n+1, n_batches))

    skip_grams = generate_skipgrams(wid, windowSize, vocab_size, negSkipgrams)
    pairs, labels = skip_grams

    # reduce amount of skip-grams
    skip_grams = np.array([list(zip(*pairs))[0], list(zip(*pairs))[1], labels], dtype='int32').T

    keep_idx = np.random.randint(low = 0, high = skip_grams.shape[0], size = int(np.ceil(skip_grams.shape[0]*keep)))
    skip_grams = skip_grams[keep_idx,]

    # save everything in one file
    f = open(file, 'ab').close() # overwrite existing files
    f = open(file, 'ab')
    np.savetxt(f, skip_grams)
    f.close()

    gc.collect()
