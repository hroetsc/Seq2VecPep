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

### INPUT ###
# data set to train bpe algorithm:
# Swissprot mouse, canonical and isoforms (reviewed proteins)
UniprotFASTA = read.fasta("./data/immunopeptidome/SwissProt_mouse_canonicalAndIsoforms.fasta", seqtype = "AA",
                          whole.header = T)
# keep track of origin of sequences
seqs = c()
origin = c()
for (e in 1:length(UniprotFASTA)) {
  seqs = c(seqs, paste(UniprotFASTA[[e]], sep = "", collapse = ""))
  origin = c(origin, getAnnot(UniprotFASTA[[e]]))
}
# merge all sequences in the fasta file
seqs = paste(seqs, sep = "", collapse = "")
write.table(seqs, "./data/immunopeptidome/concatenated_UniProt.txt", sep = "\t",
            row.names = T, col.names = T)
