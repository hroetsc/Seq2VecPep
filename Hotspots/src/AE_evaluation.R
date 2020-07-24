### HEADER ###
# HOTSPOT REGIONS
# description: use autoencoder to reduce embedding dimensionality
# input: weights file from EQ, original sequence representation
# output: atoencoded sequence representation, UMAP representations
# author: HR

library(plyr)
library(dplyr)
library(stringr)
library(rhdf5)

library(uwot)
library(ggplot2)
library(paletteer)

library(future)

### INPUT ###
weights = h5read("AE/seq2vec-TFIDF.h5", "/encoded/encoded")
View(h5ls("AE/seq2vec-TFIDF.h5"))

pre = read.csv("../RUNS/HumanProteome/word2vec_model/hp_sequence_repres_w5_d100_seq2vec-TFIDF.csv",
               stringsAsFactors = F)


### MAIN PART ###


### OUTPUT ###


