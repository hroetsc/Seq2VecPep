### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  segment antigen sequences into variable length fragments using byte-pair encoding algorithm
# input:        antigen dataset from 1_getSourceAntigens.R, mouse SwissProt sequences
# output:       encoded antigen sequences, tokens('words') for word2vec
# author:       HR

print("### TRAINING OF BYTE-PAIR ENCODING MODEL ###")

library(tibble)
library(dplyr)
library(rlist)
library(stringr)
library(seqinr)
library(berryFunctions)
library(tokenizers.bpe)

setwd("Documents/ProtTransEmbedding/Snakemake/")

### MAIN PART ###
# train BPE model
bpeModel = bpe("data/peptidome/concatenated_UniProt.txt",
               coverage = 0.999,
               vocab_size = 100000, #5000 #10000 #50000 #100000
               threads = 11,
               model_path = "results/encoded_proteome/opt_BPE_model_100000.bpe")
