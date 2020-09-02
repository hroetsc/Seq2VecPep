### HEADER ###
# HOTSPOT PREDICTION
# description: calculate embedding matrices for every window in train and test data set
# input: table with tokens in sliding windows, token embeddings, tf-idf scores
# output: big matrix files (train and test data)
# author: HR

import os
import numpy as np
import pandas as pd
import multiprocessing as mp

import C_helper

### INPUT ###
# for benchmarking
# tokensAndCounts_benchmark = pd.read_csv('data/windowTokens_benchmark.csv')

ext = ""
print('extension: ', ext)

subs = 'OPT'
print('subset: ', subs)

tokensAndCounts_train = pd.read_csv(str('data/'+ext+'windowTokens_'+subs+'training.csv'))
tokensAndCounts_test = pd.read_csv(str('data/'+ext+'windowTokens_'+subs+'testing.csv'))

weights = pd.read_csv('data/token_AAindices.csv')


### MAIN PART ###

# hyperparameters
workers = 16
embeddingDim = 128
tokPerWindow = 8

# weight_bench = 'data/embMatrices_benchmark.dat'
# acc_bench = 'data/embMatricesAcc_benchmark.dat'

weight_train = str('data/AAindex_'+ext+'embMatrices_'+subs+'training.dat')
acc_train = str('data/AAindex_'+ext+'embMatricesAcc_'+subs+'training.dat')

weight_test = str('data/AAindex_'+ext+'embMatrices_'+subs+'testing.dat')
acc_test = str('data/AAindex_'+ext+'embMatricesAcc_'+subs+'testing.dat')


# get tokens
def get_tokens(tokensAndCounts):
    # append integers to tokens to correctly identify embedding matrices later
    tokensAndCounts['idx'] = tokensAndCounts.reset_index().index
    tokens = np.array(tokensAndCounts.loc[:, ['Accession', 'tokens', 'idx']], dtype='object')

    return tokens

# tokens_bench = get_tokens(tokensAndCounts_benchmark)
tokens_train = get_tokens(tokensAndCounts_train)
tokens_test = get_tokens(tokensAndCounts_test)


pool = mp.Pool(workers)

# benchmarking
# print('embedding matrices for benchmarking data')
# if __name__ == "__main__":
#     pool.starmap( C_helper.findEmbeddings,
#                   [[batch_token, weights, embeddingDim, tokPerWindow, weight_bench, acc_bench] for batch_token in list(tokens_bench)] )

# training data
print('embedding matrices for training data')
if __name__ == "__main__":
    pool.starmap( C_helper.findEmbeddings,
                  [[batch_token, weights, embeddingDim, tokPerWindow, weight_train, acc_train] for batch_token in list(tokens_train)] )

# testing data
print('embedding matrices for testing data')
if __name__ == "__main__":
    pool.starmap( C_helper.findEmbeddings,
                  [[batch_token, weights, embeddingDim, tokPerWindow, weight_test, acc_test] for batch_token in list(tokens_test)] )

pool.close()