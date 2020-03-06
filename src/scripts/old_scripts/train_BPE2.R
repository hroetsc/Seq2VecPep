### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  segment antigen sequences into variable length fragments using bite-pair encoding algorithm
# input:        antigen dataset from 1_getSourceAntigens.R, mouse SwissProt sequences
# output:       encoded antigen sequences, tokens('words') for word2vec
# author:       HR

library(tibble)
library(dplyr)
library(rlist)
#library(Rcpi)
library(stringr)
library(seqinr)
library(berryFunctions)
library(tokenizers.bpe)

setwd("/home/hroetsc/Documents/ProtTransEmbedding/Snakemake/")

### MAIN PART ###
# train BPE model
bpeModel = bpe("./data/immunopeptidome/concatenated_UniProt.txt",
               coverage = 0.999,
               vocab_size = 50000,
               threads = 11,
               model_path = "./results/encoded_antigens/BPE_model.bpe")
