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

### MAIN PART ###
# train BPE model
bpeModel = bpe(snakemake@input[["conc_UniProt"]],
               coverage = 0.999,
               vocab_size = 10000, #50000 #10000 #1000 #python runs out of memory when generating skip_grams
               threads = 11,
               model_path = unlist(snakemake@output[["BPE_model"]]))
