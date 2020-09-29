### HEADER ###
# HOTSPOT PREDICTION
# description: helper script for parallel calculation of embedding matrices
# input: -
# output: -
# author: HR

import os
import multiprocessing as mp
import numpy as np

def findEmbeddings(batch_token, weights, embeddingDim, tokPerWindow, weightfile, accfile):
    tokens_split = [str.split(' ') for str in [batch_token[1]]][0]

    out = [None] * tokPerWindow

    for n, t in enumerate(tokens_split):
        # find embedding
        out[n] = np.array(weights[weights['subword'] == t].iloc[:, 1:(embeddingDim + 1)], dtype='float32').flatten()

    out = np.asarray(out, dtype='float32')
    idx = np.asarray(batch_token[2], dtype='int32')


    # save accessions and weight matrix as numpy array in binary format
    with open(weightfile, 'ab') as wf, open(accfile, 'ab') as af:
        out.tofile(wf)
        idx.tofile(af)

        wf.close()
        af.close()


# old version: use seq2vec + TFIDF
def findEmbeddings_tfidf(batch_token, tfidf, weights, embeddingDim, tokPerWindow, weightfile, accfile):
    tokens_split = [str.split(' ') for str in [batch_token[1]]][0]
    acc = batch_token[0]

    out = [None] * tokPerWindow

    for n, t in enumerate(tokens_split):
        # find embedding
        emb = np.array(weights[weights['subword'] == t].iloc[:, 1:(embeddingDim + 1)], dtype='float32').flatten()
        # find tf-idf
        tf_idf = float(tfidf[(tfidf['Accession'] == acc) & (tfidf['token'] == t)]['tf_idf'])

        # multiply embeddings by tf-idf score
        out[n] = tf_idf * emb

    out = np.asarray(out, dtype='float32')
    idx = np.asarray(batch_token[2], dtype='int32')

    # save accessions and weight matrix as numpy array in binary format
    with open(weightfile, 'ab') as wf, open(accfile, 'ab') as af:
        out.tofile(wf)
        idx.tofile(af)

        wf.close()
        af.close()
