### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  "true" similarity between protein sequences based on pairwise sequence alignment
#               (Needleman - Wunsch)
# input:        sequences
# output:       syntactic similarity matrix
# author:       HR

library(protr)
library(future)
library(dplyr)
library(foreach)
library(doParallel)
library(Biostrings)


print("### TRUE SYNTACTIC SIMILARITY")

### INPUT ###
# formatted sequences
seqes = read.csv(snakemake@input[["formatted_sequence"]], stringsAsFactors = F, header = T)
# seqes = read.csv("data/current_sequences.csv", stringsAsFactors = F, header = T)


### MAIN PART ###
# order accessions alphabetically
seqes = seqes[order(seqes$Accession), ]


# parallelised protein sequence similarity calculation
cores = availableCores()
alig = parSeqSim(seqes$seqs,
                 cores = cores,
                 batches = 10, # to save memory
                 verbose = T,
                 type = "local",
                 gap.opening = -2,
                 gap.extension = -8,
                 submat = "BLOSUM50")

alig = as.matrix(alig) %>% as.data.frame()


for (p in 2:ncol(alig)){
  alig[,p] = as.character(as.vector(alig[,p]))
}


res = cbind(seqes$Accession, alig)
colnames(res) = c("Accession", seq(1, ncol(alig)))

res = as.data.frame(res)


### OUTPUT ###
write.csv(res, file = unlist(snakemake@output[["syntax"]]), row.names = F)
