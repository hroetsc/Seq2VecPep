### HEADER ###
# HOTSPOT PREDICTION
# description: improve model interpretability by unraveling feature attribution
# input: model, some features
# output: SHAP values
# author: HR

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import shap

pseudocounts = 1
tokPerWindow = 8
embeddingDim = 128
batchSize = 16

no_models = 8

### INPUT ###
# load some of the data
mu = pd.read_csv('CNN/data/mean_emb.csv')
mu = np.tile(np.array(mu).flatten(), tokPerWindow).reshape((tokPerWindow, embeddingDim))

tokensAndCounts = pd.read_csv('CNN/data/windowTokens_OPTtraining.csv')
emb = 'CNN/data/embMatrices_OPTtraining.dat'
acc = 'CNN/data/embMatricesAcc_OPTtraining.dat'

tokensAndCounts_test = pd.read_csv('CNN/data/windowTokens_OPTtesting.csv')
emb_test = 'CNN/data/embMatrices_OPTtesting.dat'
acc_test = 'CNN/data/embMatricesAcc_OPTtesting.dat'

def format_input(tokensAndCounts):
    tokens = np.array(tokensAndCounts.loc[:, ['Accession', 'tokens']], dtype='object')
    counts = np.array(tokensAndCounts['counts'], dtype='float32')

    # log-transform counts (+ pseudocounts)
    counts = np.log2((counts + pseudocounts))

    print('number of features: ', counts.shape[0])
    return tokens, counts


def open_and_format_matrices(tokens, counts, emb_path, acc_path):
    no_elements = int(tokPerWindow * embeddingDim)  # number of matrix elements per sliding window
    # how many bytes are this? (32-bit encoding --> 4 bytes per element)
    chunk_size = int(no_elements * 4)

    embMatrix = [None] * tokens.shape[0]
    accMatrix = [None] * tokens.shape[0]
    chunk_pos = 0

    # open weights and accessions binary file
    with open(emb_path, 'rb') as emin, open(acc_path, 'rb') as ain:
        # loop over files to get elements
        for b in range(tokens.shape[0]):
            emin.seek(chunk_pos, 0)  # set cursor position with respect to beginning of file
            # read current chunk of embeddings and format in matrix shape
            dt = np.fromfile(emin, dtype='float32', count=no_elements)

            # make sure to pass 4D-Tensor to model: (batchSize, depth, height, width)
            dt = dt.reshape((tokPerWindow, embeddingDim))
            # dt = dt - mu
            embMatrix[b] = np.expand_dims(dt, axis=0)

            # get current accession (index)
            ain.seek(int(b * 4), 0)
            accMatrix[b] = int(np.fromfile(ain, dtype='int32', count=1))

            # increment chunk position
            chunk_pos += chunk_size

        emin.close()
        ain.close()

    # order tokens and count according to order in embedding matrix
    accMatrix = np.array(accMatrix, dtype='int32')
    tokens = tokens[accMatrix, :]
    counts = counts[accMatrix]

    embMatrix = np.array(embMatrix, dtype='float32')

    # output: reformatted tokens and counts, embedding matrix
    return tokens, counts, embMatrix

tokens, counts = format_input(tokensAndCounts)
tokens, counts, emb = open_and_format_matrices(tokens, counts, emb, acc)

tokens_test, counts_test = format_input(tokensAndCounts_test)
tokens_test, counts_test, emb_test = open_and_format_matrices(tokens_test, counts_test, emb_test, acc_test)


### MAIN PART ###
### visualize embeddings ###
# pick 200 random embeddings and their counts
idx = np.random.randint(0, emb.shape[0], 200)
tmp_counts = [None]*len(idx)
tmp_embs = [None]*len(idx)

for i in range(len(idx)):
    tmp_counts[i] = float(counts[idx[i]])
    tmp_embs[i] = emb[idx[i], :, :, :].reshape(tokPerWindow, embeddingDim)

tmp_counts = np.array(tmp_counts)
tmp_embs = np.array(tmp_embs)

# sort embeddings by counts
sort = np.argsort(tmp_counts)
tmp_counts = tmp_counts[sort]
tmp_embs = tmp_embs[sort]

# plot
fig, axs = plt.subplots(nrows=len(idx), ncols=1, figsize=(1, 12))
for i in range(len(sort)):
    axs[i].imshow(tmp_embs[i, :, :])
    axs[i].axis('off')
fig.tight_layout()
plt.savefig(str('CNN/results/sequence_matrices/sequence_sorted.pdf'), dpi=1200)
plt.show()

pca = PCA(n_components=8)
fig, axs = plt.subplots(nrows=len(idx), ncols=1, figsize=(1, 12))
for i in range(len(sort)):
    axs[i].imshow(pca.fit_transform(tmp_embs[i, :, :]))
    axs[i].axis('off')
fig.tight_layout()
plt.savefig(str('CNN/results/sequence_matrices/sequence_PCA.pdf'), dpi=1200)
plt.show()


### load models and create emsemble
all_models = [None] * no_models

for i in range(no_models):
    print('loading model from rank ', i)

    model = tf.keras.models.load_model('CNN/results/model/best_model_rank{}.h5'.format(i))

    for layer in model.layers:
        layer.trainable = False
        layer._name = 'ensemble_' + str(i) + '_' + layer.name

    all_models[i] = model

print('building ensemble')
ensemble_in = [model.input for model in all_models]
ensemble_out = [model.output for model in all_models]

merge = layers.Concatenate()(ensemble_out)
dense = layers.Dense(128, activation='relu',
                     kernel_initializer=tf.keras.initializers.HeNormal())(merge)
out = layers.Dense(1, activation='linear',
                   name='ensemble_output')(dense)

ensemble = keras.Model(inputs=ensemble_in, outputs=out)
ensemble.compile(loss='mean_squared_error',
                 metrics='mean_absolute_error',
                 optimizer=keras.optimizers.Adam(learning_rate=0.001))

ensemble.summary()

ensemble.fit(x=emb,
             y=counts,
             batch_size=batchSize,
             validation_data=(emb_test, counts_test),
             validation_batch_size=batchSize,
             epochs=10,
             initial_epoch=1,
             max_queue_size=256,
             verbose=2,
             shuffle=True)

# save ensemble
ensemble.save('CNN/results/ensemble.h5')

### MAIN PART ###
# make prediction
pred = ensemble.predict(x=emb_test,
                        batch_size=16,
                        verbose=1,
                        max_queue_size=256)
print('counts:')
print(counts_test)

print('prediction:')
print(pred.flatten())

# merge actual and predicted counts
prediction = pd.DataFrame({"Accession": tokens_test[:, 0],
                           "window": tokens_test[:, 1],
                           "count": counts_test,
                           "pred_count": pred.flatten()})

print('SAVE PREDICTED COUNTS')
pd.DataFrame.to_csv(prediction,
                    'CNN/results/ensemble_predictions.csv',
                    index=False)


### apply shap values
background = emb[np.random.choice(emb.shape[0], 100, replace=False), :, :, :]
e = shap.DeepExplainer(ensemble, background)

shap_values = e.shap_values(background)
