### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  embedding based on biophysical properties
#               (removing properties correlating with sequence length)
# input:        sequences
# output:       sequence embedding using biophysical properties
# author:       HR

library(seqinr)
library(protr)
library(Peptides)
library(plyr)
library(dplyr)
library(stringr)
library(readr)


print("### BIOPHYSICAL PROPERTY EMBEDDINGS ###")

### INPUT ###
# formatted sequences
sequences = read.csv(snakemake@input[["formatted_sequence"]], stringsAsFactors = F, header = T)
Props = read.csv(snakemake@input[["seq_props"]], stringsAsFactors = F, header = T)

### MAIN PART ###
sequences = sequences[order(sequences$seqs), ]

PropMatrix = Props[which(Props$seqs %in% sequences$seqs), ]
PropMatrix = PropMatrix[order(PropMatrix$seqs), ]

sequences = sequences[which(sequences$seqs %in% PropMatrix$seqs), ]

PropMatrix = cbind(sequences$Accession, PropMatrix)
colnames(PropMatrix)[1] = "Accession"
PropMatrix$X = NULL

### OUTPUT ###
write.csv(x = PropMatrix, file = unlist(snakemake@output[["embedding_biophys"]]), row.names = F)