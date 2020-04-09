### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  embedding based on autocorrelation of dinucleotides (BioMedR package)
# input:        sequences
# output:       embedded sequences
# author:       HR

library(seqinr)
library(plyr)
library(dplyr)
library(stringr)
library(readr)
library(BioMedR)


library(parallel)
library(foreach)
library(doParallel)
library(doMC)
library(plyr)

cl <- makeCluster(detectCores())
registerDoParallel(cl)
registerDoMC(detectCores())


### INPUT ###
# formatted sequences
#sequences = read.csv("transcriptome/data/red_transcriptome_human.csv", stringsAsFactors = F, header = T)
sequences = read.csv(snakemake@input[["formatted_sequence"]], stringsAsFactors = F, header = T)


### MAIN PART ###
lag = min(nchar(sequences$seqs)) -2
if(lag > 84){
  lag = 84
}

#DNAPse.master = matrix(ncol = 16+lag, nrow = nrow(sequences))

DNAPse.master = foreach(i = 1:nrow(sequences), .combine = "rbind") %dopar% {
  extrDNAPseDNC(sequences$seqs[i], lambda = lag)
}

# clean
master = as.matrix(DNAPse.master)
DNAPse.master[which(!is.finite(DNAPse.master))] = 0

master = cbind(sequences, DNAPse.master)

### OUTPUT ###
write.csv(x = master, file = unlist(snakemake@output[["embedding_DNAPse"]]), row.names = F)
