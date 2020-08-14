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
# tokensAndCounts_benchmark = pd.read_csv('results/windowTokens_benchmark.csv')
tokensAndCounts_train = pd.read_csv('data/windowTokens_OPTtraining.csv')
tokensAndCounts_test = pd.read_csv('data/windowTokens_OPTtesting.csv')

weights = pd.read_csv('data/token_embeddings.csv')

tfidf_train = pd.read_csv('data/TFIDF_training.csv')
tfidf_test = pd.read_csv('data/TFIDF_testing.csv')


### MAIN PART ###

# hyperparameters
workers = 72
embeddingDim = 128
tokPerWindow = 8

# weight_bench = 'results/embMatrices_benchmark.dat'
# acc_bench = 'results/embMatricesAcc_benchmark.dat'

weight_train = 'data/embMatrices_training.dat'
acc_train = 'data/embMatricesAcc_training.dat'

weight_test = 'data/embMatrices_testing.dat'
acc_test = 'data/embMatricesAcc_testing.dat'


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
print('embedding matrices for benchmarking data')
# if __name__ == "__main__":
#     pool.starmap( C_helper.findEmbeddings,
#                   [[batch_token, tfidf_train, weights, embeddingDim, tokPerWindow, weight_bench, acc_bench] for batch_token in list(tokens_bench)] )

# training data
print('embedding matrices for training data')
if __name__ == "__main__":
    pool.starmap( C_helper.findEmbeddings,
                  [[batch_token, tfidf_train, weights, embeddingDim, tokPerWindow, weight_train, acc_train] for batch_token in list(tokens_train)] )

# testing data
print('embedding matrices for testing data')
if __name__ == "__main__":
    pool.starmap( C_helper.findEmbeddings,
                  [[batch_token, tfidf_test, weights, embeddingDim, tokPerWindow, weight_test, acc_test] for batch_token in list(tokens_test)] )

pool.close()