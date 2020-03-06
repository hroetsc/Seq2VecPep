print("### PROTEOME EMBEDDING USING SKIP-GRAM NEURAL NETWORK - 1 ###")
import os
import sys
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

# import scripts for hyperparameter optimization
import skopt
from skopt import gbrt_minimize, gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer

def generate_skipgrams(wid, windowSize):
    skip_grams = skipgrams(sequence = wid,
                                vocabulary_size = len(wid)+1,
                                window_size = windowSize,
                                categorical = False,
                                shuffle = False,
                                negative_samples = 0.5)
    gc.collect()
    return skip_grams


def parallel_processing(n, wid, windowSize, n_batches, tars, cons, labs):
    #print("generating batch {} of {}".format(n+1, n_batches))

    skip_grams = generate_skipgrams(wid, windowSize)
    pairs, labels = skip_grams

    # save everything
    tar = open(tars, 'ab').close() # overwrite existing files
    tar = open(tars, 'ab')
    np.savetxt(tar, np.array(list(zip(*pairs))[0], dtype='int32'))
    tar.close()

    con = open(cons, 'ab').close()
    con = open(cons, 'ab')
    np.savetxt(con, np.array(list(zip(*pairs))[1], dtype='int32'))
    con.close()

    lab = open(labs, 'ab').close()
    lab = open(labs, 'ab')
    np.savetxt(lab, np.array(labels, dtype='int32'))
    lab.close()

    gc.collect()
