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
from sklearn.feature_selection import *
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

proteins = pd.read_csv('CNN/data/sequence_embedings.csv')

def format_input(tokensAndCounts):
    tokens = np.array(tokensAndCounts.loc[:, ['Accession', 'tokens']], dtype='object')
    counts = np.array(tokensAndCounts['counts'], dtype='float32')

    # log-transform counts (+ pseudocounts)
    counts = np.log2((counts + pseudocounts))

    # get binary labels
    labels = np.where(counts == 0, 0, 1)

    print('number of features: ', counts.shape[0])

    return tokens, labels, counts



def open_and_format_matrices(tokens, labels, counts, emb_path, acc_path, augment=False, ret_pca=False):
    no_elements = int(tokPerWindow * embeddingDim)  # number of matrix elements per sliding window
    # how many bytes are this? (32-bit encoding --> 4 bytes per element)
    chunk_size = int(no_elements * 4)

    embMatrix = [None] * tokens.shape[0]
    PCAs = [None] * tokens.shape[0]
    accMatrix = [None] * tokens.shape[0]
    chunk_pos = 0

    pca = PCA(n_components=tokPerWindow)

    def scaling(x, a, b):
        sc_x = (x - np.min(x)) / (np.max(x) - np.min(x))
        return (b - a) * sc_x + a

    # open weights and accessions binary file
    with open(emb_path, 'rb') as emin, open(acc_path, 'rb') as ain:
        # loop over files to get elements
        for b in range(tokens.shape[0]):
            emin.seek(chunk_pos, 0)  # set cursor position with respect to beginning of file
            # read current chunk of embeddings and format in matrix shape
            dt = np.fromfile(emin, dtype='float32', count=no_elements)

            # scale input data between 0 and 255 (color)
            dt = scaling(dt, a=0, b=255)

            # make sure to pass 4D-Tensor to model: (batchSize, depth, height, width)
            dt = dt.reshape((tokPerWindow, embeddingDim))

            # apply PCA
            PCAs[b] = pca.fit_transform(dt)

            # for 2D convolution --> 5d input:
            # embMatrix[b] = np.expand_dims(dt, axis=0)
            embMatrix[b] = dt

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
        labels = labels[accMatrix]

        embMatrix = np.array(embMatrix, dtype='float32')
        PCAs = np.array(PCAs)

        # randomly augment training data
        if augment:
            rnd_size = int(np.ceil(embMatrix.shape[0] * 0.15))
            rnd_vflip = np.random.randint(0, embMatrix.shape[0], rnd_size)
            rnd_hflip = np.random.randint(0, embMatrix.shape[0], rnd_size)
            rnd_bright = np.random.randint(0, embMatrix.shape[0], rnd_size)

            embMatrix_vflip = np.flipud(embMatrix[rnd_vflip, :, :, :])
            embMatrix_hflip = np.fliplr(embMatrix[rnd_hflip, :, :, :])
            embMatrix_bright = embMatrix[rnd_bright, :, :, :] + np.random.uniform(-1, 1)

            embMatrix = np.concatenate((embMatrix, embMatrix_vflip, embMatrix_hflip, embMatrix_bright))
            PCAs = np.concatenate((PCAs, PCAs[rnd_vflip], PCAs[rnd_hflip], PCAs[rnd_bright]))
            labels = np.concatenate((labels, labels[rnd_vflip], labels[rnd_hflip], labels[rnd_bright]))
            counts = np.concatenate((counts, counts[rnd_vflip], counts[rnd_hflip], counts[rnd_bright]))
            tokens = np.concatenate((tokens, tokens[rnd_vflip], tokens[rnd_hflip], tokens[rnd_bright]))

        # output: reformatted tokens and counts, embedding matrix
        if ret_pca:
            return tokens, labels, counts, embMatrix, PCAs

        else:
            return tokens, labels, counts, embMatrix

tokens, labels, counts = format_input(tokensAndCounts)
tokens, labels, counts, emb, pca = open_and_format_matrices(tokens, labels, counts, emb, acc, ret_pca=True)

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

### feature selection ###
# for embedding + convolution --> test if dimensions are unique for all tokens
fst_dim = emb[:, 0, 0, 0]
fst_dim_un = np.unique(fst_dim)

tmp = np.array([str(tokens[i, 1]).split() for i in range(tokens.shape[0])])
tmp2 = np.unique(tmp.flatten())

# transform embeddings to select features
emb_flat = np.array([emb[i, :, :].flatten() for i in range(emb.shape[0])])

# based on f-value
freg_selector = SelectKBest(f_regression)
freg_features = freg_selector.fit_transform(X=emb_flat, y=counts)
freg_scores = -np.log10(freg_selector.pvalues_)

plt.figure()
plt.suptitle('feature importance in test data set')
plt.title('based on regression F-value')
plt.xlabel('# feature')
plt.ylabel('-log10 selector p-value')
plt.bar(np.arange(emb_flat.shape[1]), freg_scores)
plt.savefig('CNN/feature_importance_regFvalue.png', dpi=300)
plt.show()


fclass_selector = SelectKBest(f_classif)
fclass_features = fclass_selector.fit_transform(X=emb_flat, y=labels)
fclass_scores = -np.log10(fclass_selector.pvalues_)

plt.figure()
plt.suptitle('feature importance in test data set')
plt.title('based on classification F-value')
plt.xlabel('# feature')
plt.ylabel('-log10 selector p-value')
plt.bar(np.arange(emb_flat.shape[1]), fclass_scores)
plt.savefig('CNN/feature_importance_classFvalue.png', dpi=300)
plt.show()

# based on mutual information
mi_selector = SelectKBest(mutual_info_regression)
mi_features = mi_selector.fit_transform(X=emb_flat, y=counts)


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
