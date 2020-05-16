### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  "true" similarity between protein sequences based on pairwise sequence alignment
#               (Needleman - Wunsch)
# input:        sequences
# output:       syntactic similarity matrix
# author:       HR

library(protr)
library(future)

print("### TRUE SYNTACTIC SIMILARITY")

### INPUT ###
# formatted sequences
sequences = read.csv(snakemake@input[["formatted_sequence"]], stringsAsFactors = F, header = T)
# sequences = read.csv("proteome/data/red_formatted_proteome.csv", stringsAsFactors = F, header = T)


### MAIN PART ###
# order accessions alphabetically
sequences = sequences[order(sequences$Accession), ]


# parallelised protein sequence similarity calculation
cores = availableCores()
alig = parSeqSim(sequences$seqs,
                 cores = cores,
                 batches = 30, # to save memory
                 verbose = T,
                 type = "local",
                 gap.opening = -2,
                 gap.extension = -8,
                 submat = "BLOSUM50")


res = matrix(ncol = ncol(alig)+1, nrow = nrow(alig))
res[, 1] = sequences$Accession
res[, c(2:ncol(res))] = alig
colnames(res) = c("Accession", seq(1, ncol(alig)))

res = as.data.frame(res)

for (p in 2:ncol(res)){
  res[,p] = as.numeric(as.character(res[,p]))
}

### OUTPUT ###
write.csv(res, file = unlist(snakemake@output[["syntax"]]), row.names = T)
