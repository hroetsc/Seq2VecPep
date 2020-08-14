### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  train byte-pair encoding algorithm
# input:        concatenated sequences
# output:       BPE model
# author:       HR

print("### TRAINING OF BYTE-PAIR ENCODING MODEL ###")

library(tibble)
library(dplyr)
library(rlist)
library(stringr)
library(seqinr)
library(berryFunctions)
library(tokenizers.bpe)


### INPUT ###
# vocab_size = 5e03
vocab_size = 5e04
threads = future::availableCores()

# load arguments
params = read.csv(snakemake@input[["params"]], stringsAsFactors = F, header = T)

vocab_size = as.numeric(params[which(params$parameter == "BPEvocab"), "value"])
print(paste0("using vocabulary size of ", vocab_size))

threads = as.numeric(params[which(params$parameter == "threads"), "value"])

# can run OOM if trained on too many threads
if(threads > 2){
  threads = 2
}


### MAIN PART ###
# train BPE model and save it
bpeModel = bpe(snakemake@input[["conc_UniProt"]],
               coverage = 0.999,
               vocab_size = vocab_size,
               threads = threads,
               model_path = unlist(snakemake@output[["BPE_model"]]))


# bpeModel = bpe("Seq2Vec/data/concatenated_UniProt_hp.txt",
#                coverage = 0.999,
#                vocab_size = vocab_size,
#                threads = threads,
#                model_path = "Seq2Vec/results/encoded_sequence/BPE_model_hp.bpe")

